#  Backend codegen for `rms_reduce`: one warp per row computes rrms[r] =
#  rsqrt(mean(x[r,:]^2)+eps) with 128-bit vectorized loads and an fp32 warp reduction, and
#  writes a single fp16 scalar per row. This is the RMSNorm reduction WITHOUT the [rows,C]
#  write of the normalized activation -- the scale is applied downstream (see rms_reduce op).
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
// One WARP per row: 128-bit (uint4=8 half) loads, fp32 sum(x^2), warp-shuffle reduction.
__global__ void {{func_name}}_kernel(const __half* __restrict__ x, __half* __restrict__ rrms,
                                     long long rows, int C, float eps) {
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const int VECS = C >> 3;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * (long long)C);
  float ss = 0.f;
  for (int v = lane; v < VECS; v += 32) {
    uint4 q = xr[v];
    const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f = __half22float2(h[i]);
      ss += f.x * f.x + f.y * f.y;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) rrms[row] = __float2half(rsqrtf(ss / (float)C + eps));
}
}  // namespace

// rrms[rows] = rsqrt(mean(x^2)+eps) over the last dim C (C % 8 == 0). No workspace.
void {{func_name}}(const void* x_ptr, void* rrms_ptr, int64_t rows, int64_t C, float eps,
                   cudaStream_t stream) {
  constexpr int BLK = 128;  // 4 warps/CTA, one row per warp
  const unsigned int grid = (unsigned int)((rows + (BLK >> 5) - 1) / (BLK >> 5));
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<__half*>(rrms_ptr),
      (long long)rows, (int)C, eps);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, void*, int64_t, int64_t, float, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{rrms_ptr}}, {{rows_expr}}, {{c}}, {{eps}}f, stream
{{indent}});
"""
)


@registry.reg("cuda.rms_reduce.gen_function")
def rms_reduce_gen_function(func_attrs):
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.rms_reduce.func_decl")
def rms_reduce_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.rms_reduce.func_call")
def rms_reduce_gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    rrms = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    c = x_shape[-1]._attrs["values"][0]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        rrms_ptr=rrms._attrs["name"],
        rows_expr=rows_expr,
        c=c,
        eps=func_attrs["eps"],
    )
