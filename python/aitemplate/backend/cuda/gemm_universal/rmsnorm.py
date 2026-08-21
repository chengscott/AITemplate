#  Backend codegen for the native `rmsnorm` op: a single-pass, one-block-per-row fused
#  RMSNorm kernel (Z = RMSNorm(X) * gamma). This ports te.rmsnorm_fwd's fusion into
#  AITemplate as inline codegen -- no libtransformer_engine link, and it compiles in the
#  same nvcc pass as the surrounding cutlass graph. fp32 accumulation matches the
#  reference (x.float().pow(2).mean().rsqrt()); the (x*rrms)->half->*gamma order matches
#  net_pt's `(x.float()*r).type_as(x) * weight`.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
// One WARP per row (ports TE's rmsnorm_fwd_tuned strategy for C<=256*8): 128-bit vectorized
// loads (uint4 = 8 halfs), fp32 sum(x^2), warp-shuffle reduction, vectorized store. C must
// be a multiple of 8. Idle lanes (v >= C/8) contribute 0 to the reduction.
__global__ void {{func_name}}_kernel(const __half* __restrict__ x,
                                     const __half* __restrict__ gamma,
                                     __half* __restrict__ z, long long rows, int C,
                                     float eps) {
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const int VECS = C >> 3;  // uint4 chunks per row (8 halfs each)
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * (long long)C);
  const uint4* gr = reinterpret_cast<const uint4*>(gamma);
  uint4* zr = reinterpret_cast<uint4*>(z + row * (long long)C);

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
  const float rrms = rsqrtf(ss / (float)C + eps);

  for (int v = lane; v < VECS; v += 32) {
    uint4 q = xr[v];
    uint4 g = gr[v];
    __half2* h = reinterpret_cast<__half2*>(&q);
    const __half2* gh = reinterpret_cast<const __half2*>(&g);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f = __half22float2(h[i]);
      float2 gf = __half22float2(gh[i]);
      float a = f.x * rrms * gf.x;
      float b = f.y * rrms * gf.y;
      {{relu_stmt}}
      h[i] = __float22half2_rn(make_float2(a, b));
    }
    zr[v] = q;
  }
}
}  // namespace

// Z[rows,C] = RMSNorm(X[rows,C]) * gamma[C]. No workspace (single pass). C % 8 == 0.
void {{func_name}}(const void* x_ptr, const void* gamma_ptr, void* z_ptr,
                   int64_t rows, int64_t C, float eps, cudaStream_t stream) {
  constexpr int BLK = 128;  // 4 warps/CTA, one row per warp
  const unsigned int grid = (unsigned int)((rows + (BLK >> 5) - 1) / (BLK >> 5));
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const __half*>(x_ptr),
      reinterpret_cast<const __half*>(gamma_ptr),
      reinterpret_cast<__half*>(z_ptr), (long long)rows, (int)C, eps);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(const void*, const void*, void*, int64_t, int64_t, float, cudaStream_t);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{gamma_ptr}}, {{z_ptr}},
{{indent}}    {{rows_expr}}, {{c}}, {{eps}}f, stream
{{indent}});
"""
)


@registry.reg("cuda.rmsnorm.gen_function")
def rmsnorm_gen_function(func_attrs):
    # optional fused relu on the output (norm_p/norm_q -> relu -> gemm): folds the relu
    # into the norm kernel so it isn't a standalone elementwise pass.
    relu_stmt = "a = fmaxf(a, 0.f); b = fmaxf(b, 0.f);" if func_attrs.get("relu") else ""
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"], relu_stmt=relu_stmt)


@registry.reg("cuda.rmsnorm.func_decl")
def rmsnorm_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.rmsnorm.func_call")
def rmsnorm_gen_function_call(func_attrs, indent="  "):
    x, gamma = func_attrs["inputs"]
    z = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    c = x_shape[-1]._attrs["values"][0]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        gamma_ptr=gamma._attrs["name"],
        z_ptr=z._attrs["name"],
        rows_expr=rows_expr,
        c=c,
        eps=func_attrs["eps"],
    )
