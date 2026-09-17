#  Backend codegen for `rms_quantize`: one warp per row reads the RAW activation x ONCE
#  (holding it in NCHUNK=ceil((C/8)/32) uint4 registers), warp-reduces BOTH sum(x^2) (-> rrms)
#  AND absmax(x) (-> per-row e4m3 scale), then writes xq (e4m3), scale (f32), rrms (f16). This
#  fuses rms_reduce + quantize_to_fp8 into a single read of x for the fp8 qkv/fc1 sub-blocks.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {
// One WARP per row: NCHUNK uint4 held in registers, fp32 warp-reduce of sum(x^2) and absmax.
__global__ void {{func_name}}_kernel(const __half* __restrict__ x, __nv_fp8_e4m3* __restrict__ xq,
                                     float* __restrict__ scale, __half* __restrict__ rrms,
                                     long long rows, int C, float eps) {
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const int VECS = C >> 3;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * (long long)C);
  uint4 xreg[{{NCHUNK}}]; float ss = 0.f, amax = 0.f;
#pragma unroll
  for (int c = 0; c < {{NCHUNK}}; c++) {
    const int v = lane + c * 32;
    if (v < VECS) {
      uint4 q = xr[v];
      const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __half22float2(h[i]);
        ss += f.x * f.x + f.y * f.y;
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
      xreg[c] = q;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  }
  const float sc = amax * (1.0f / 448.0f);
  const float inv = sc > 0.f ? (1.0f / sc) : 0.f;
  if (lane == 0) {
    rrms[row] = __float2half(rsqrtf(ss / (float)C + eps));
    scale[row] = sc > 0.f ? sc : 1.0f;
  }
  uint2* outr = reinterpret_cast<uint2*>(xq + row * (long long)C);
#pragma unroll
  for (int c = 0; c < {{NCHUNK}}; c++) {
    const int v = lane + c * 32;
    if (v < VECS) {
      const __half2* zh = reinterpret_cast<const __half2*>(&xreg[c]);
      uint2 out; unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __half22float2(zh[i]);
        f.x *= inv; f.y *= inv;
        os[i] = __nv_fp8x2_e4m3(f).__x;
      }
      outr[v] = out;
    }
  }
}
}  // namespace

// x [rows,C] f16 -> xq [rows,C] e4m3 + scale [rows] f32 + rrms [rows] f16. C % 8 == 0.
void {{func_name}}(const void* x_ptr, void* xq_ptr, void* scale_ptr, void* rrms_ptr,
                   int64_t rows, int64_t C, float eps, cudaStream_t stream) {
  constexpr int BLK = 128;  // 4 warps/CTA, one row per warp
  const unsigned int grid = (unsigned int)((rows + (BLK >> 5) - 1) / (BLK >> 5));
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<__nv_fp8_e4m3*>(xq_ptr),
      reinterpret_cast<float*>(scale_ptr), reinterpret_cast<__half*>(rrms_ptr),
      (long long)rows, (int)C, eps);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, void*, void*, void*, int64_t, int64_t, float, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{xq_ptr}}, {{scale_ptr}}, {{rrms_ptr}}, {{rows_expr}}, {{c}}, {{eps}}f, stream
{{indent}});
"""
)


def _C(func_attrs):
    return func_attrs["inputs"][0]._attrs["shape"][-1]._attrs["values"][0]


@registry.reg("cuda.rms_quantize.gen_function")
def gen_function(func_attrs):
    C = _C(func_attrs)
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"], NCHUNK=((C >> 3) + 31) // 32)


@registry.reg("cuda.rms_quantize.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.rms_quantize.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xq, scale, rrms = func_attrs["outputs"]
    rows_expr = " * ".join(d._attrs["name"] for d in x._attrs["shape"][:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent, func_name=func_attrs["name"],
        x_ptr=x._attrs["name"], xq_ptr=xq._attrs["name"],
        scale_ptr=scale._attrs["name"], rrms_ptr=rrms._attrs["name"],
        rows_expr=rows_expr, c=_C(func_attrs), eps=func_attrs["eps"],
    )
