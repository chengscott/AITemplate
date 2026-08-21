#  Backend codegen for `rope`: one fused kernel for the learnable-2D RoPE rotation with
#  baked cos/sin -- collapses the split + mul/sub/add + concatenate chain into a single
#  kernel (TE fused_rope in spirit; ours is 2D-learnable with baked cos/sin, interleaved
#  pairs). X:[B,S,H,D] (pairs (2p,2p+1)), cos/sin:[1,S,H,P] (P=D/2) -> rotated [B,S,H,D].
import jinja2

from aitemplate.backend import registry

# scalar: one thread per (row, p) pair. General over any P.
SCALAR_FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
// One thread per (row, p) pair; row = b*S*H + s*H + h. cos/sin broadcast over batch, so
// index = (row % (S*H))*P + p. out[2p]=x0*c - x1*s; out[2p+1]=x0*s + x1*c.
__global__ void {{func_name}}_kernel(const __half* __restrict__ x,
                                     const __half* __restrict__ cosb,
                                     const __half* __restrict__ sinb,
                                     __half* __restrict__ out, long long rows) {
  const int SH = {{sh}}, P = {{p}};
  const long long total = rows * (long long)P;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long long row = i / P;
  const int p = (int)(i - row * P);
  const __half* xr = x + row * (2LL * P);
  __half* orow = out + row * (2LL * P);
  const long long ci = (long long)(row % SH) * P + p;
  const float x0 = __half2float(xr[2 * p]);
  const float x1 = __half2float(xr[2 * p + 1]);
  const float c = __half2float(cosb[ci]);
  const float s = __half2float(sinb[ci]);
  orow[2 * p] = __float2half(x0 * c - x1 * s);
  orow[2 * p + 1] = __float2half(x0 * s + x1 * c);
}
}  // namespace

// out[B,S,H,D] = RoPE(x, cos, sin). No workspace.
void {{func_name}}(const void* x, const void* cosb, const void* sinb, void* out,
                   int64_t rows, cudaStream_t stream) {
  const int P = {{p}};
  const long long total = rows * (long long)P;
  const int block = 256;
  const unsigned int grid = (unsigned int)((total + block - 1) / block);
  {{func_name}}_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(cosb),
      reinterpret_cast<const __half*>(sinb), reinterpret_cast<__half*>(out),
      (long long)rows);
}
"""
)

# vectorized: one thread per HEAD, specialized for P==8 (head_dim=16). x/out = 2 uint4,
# cos/sin = 1 uint4, all 128-bit and register-resident. ~1.8x over scalar (bit-exact).
VEC8_FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
__global__ void {{func_name}}_kernel(const __half* __restrict__ x,
                                     const __half* __restrict__ cosb,
                                     const __half* __restrict__ sinb,
                                     __half* __restrict__ out, long long rows) {
  const int SH = {{sh}};  // P == 8
  const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * 16LL);   // 2*P=16 halfs = 2 uint4
  uint4* orow = reinterpret_cast<uint4*>(out + row * 16LL);
  const long long cbase = (long long)(row % SH) * 8;
  const uint4 cq = reinterpret_cast<const uint4*>(cosb + cbase)[0];   // 8 cos
  const uint4 sq = reinterpret_cast<const uint4*>(sinb + cbase)[0];   // 8 sin
  const __half* ch = reinterpret_cast<const __half*>(&cq);
  const __half* sh = reinterpret_cast<const __half*>(&sq);
  uint4 q0 = xr[0], q1 = xr[1];
  __half2* xh0 = reinterpret_cast<__half2*>(&q0);
  __half2* xh1 = reinterpret_cast<__half2*>(&q1);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float2 xp = __half22float2(xh0[k]);
    float c = __half2float(ch[k]), s = __half2float(sh[k]);
    xh0[k] = __float22half2_rn(make_float2(xp.x * c - xp.y * s, xp.x * s + xp.y * c));
  }
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float2 xp = __half22float2(xh1[k]);
    float c = __half2float(ch[4 + k]), s = __half2float(sh[4 + k]);
    xh1[k] = __float22half2_rn(make_float2(xp.x * c - xp.y * s, xp.x * s + xp.y * c));
  }
  orow[0] = q0; orow[1] = q1;
}
}  // namespace

// out[B,S,H,D] = RoPE(x, cos, sin), one thread per head (rows = B*S*H). No workspace.
void {{func_name}}(const void* x, const void* cosb, const void* sinb, void* out,
                   int64_t rows, cudaStream_t stream) {
  const int block = 256;
  const unsigned int grid = (unsigned int)((rows + block - 1) / block);
  {{func_name}}_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(cosb),
      reinterpret_cast<const __half*>(sinb), reinterpret_cast<__half*>(out),
      (long long)rows);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, const void*, const void*, void*, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x}}, {{cos}}, {{sin}}, {{out}}, {{rows_expr}}, stream
{{indent}});
"""
)


@registry.reg("cuda.rope2d.gen_function")
def rope_gen_function(func_attrs):
    # P==8 (head_dim=16, the go9/go19 deploy case) -> all-128-bit thread-per-head kernel;
    # any other P -> general scalar kernel.
    tmpl = VEC8_FUNC_TEMPLATE if func_attrs["p"] == 8 else SCALAR_FUNC_TEMPLATE
    return tmpl.render(
        func_name=func_attrs["name"], sh=func_attrs["sh"], p=func_attrs["p"]
    )


@registry.reg("cuda.rope2d.func_decl")
def rope_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.rope2d.func_call")
def rope_gen_function_call(func_attrs, indent="  "):
    x, cosb, sinb = func_attrs["inputs"]
    out = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]  # [B,S,H,D]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x=x._attrs["name"],
        cos=cosb._attrs["name"],
        sin=sinb._attrs["name"],
        out=out._attrs["name"],
        rows_expr=rows_expr,
    )
