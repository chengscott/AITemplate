#  Backend codegen for `rms_quantize_mxfp8`: one warp per row reads raw x ONCE; each lane owns
#  32-element K-blocks, computing per-block amax -> ue8m0 block scale (round-up-pow2(amax/448))
#  + e4m3 data (written to the SWIZZLED SFA buffer via gemm_swizzled_scale_idx), and accumulating
#  sum(x^2) for the per-token rrms (warp-reduced). Emits xq (e4m3), sfa (ue8m0 swizzled), rrms.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>

namespace {
// warp per row; lane owns 32-elem K-blocks (nkb of them). sfa swizzled (128x4 cuBLAS layout).
__global__ void {{func_name}}_kernel(const __half* __restrict__ x, unsigned char* __restrict__ xq,
                                     unsigned char* __restrict__ sfa, __half* __restrict__ rrms,
                                     long long rows, int C, int nkb, int ntx, float eps) {
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * (long long)C);   // 8 half / uint4
  uint2* xqr = reinterpret_cast<uint2*>(xq + row * (long long)C);             // 8 e4m3 / uint2
  const long long iy = row & 127;
  float ss = 0.f;
  for (int kb = lane; kb < nkb; kb += 32) {
    uint4 buf[4]; float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint4 q = xr[kb * 4 + j]; buf[j] = q;
      const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); ss += f.x * f.x + f.y * f.y; amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y))); }
    }
    int b = amax > 0.f ? ((int)ceilf(__log2f(amax * (1.0f / 448.0f))) + 127) : 0;
    b = b < 0 ? 0 : (b > 254 ? 254 : b);
    float inv = amax > 0.f ? exp2f((float)(127 - b)) : 0.f;
    sfa[(row >> 7) * (long long)ntx * 512 + (kb >> 2) * 512 + (iy % 32) * 16 + (iy >> 5) * 4 + (kb & 3)] = (unsigned char)b;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const __half2* h = reinterpret_cast<const __half2*>(&buf[j]);
      uint2 out; unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); f.x *= inv; f.y *= inv; os[i] = __nv_fp8x2_e4m3(f).__x; }
      xqr[kb * 4 + j] = out;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) rrms[row] = __float2half(rsqrtf(ss / (float)C + eps));
}
}  // namespace

// x [rows,C] f16 -> xq [rows,C] e4m3 + sfa (swizzled ue8m0) + rrms [rows] f16. C % 32 == 0.
void {{func_name}}(const void* x_ptr, void* xq_ptr, void* sfa_ptr, void* rrms_ptr,
                   int64_t rows, int64_t C, float eps, cudaStream_t stream) {
  const int nkb = (int)(C / 32), ntx = (nkb + 3) / 4;
  constexpr int BLK = 128;
  const unsigned int grid = (unsigned int)((rows + (BLK >> 5) - 1) / (BLK >> 5));
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<unsigned char*>(xq_ptr),
      reinterpret_cast<unsigned char*>(sfa_ptr), reinterpret_cast<__half*>(rrms_ptr),
      (long long)rows, (int)C, nkb, ntx, eps);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, void*, void*, void*, int64_t, int64_t, float, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{xq_ptr}}, {{sfa_ptr}}, {{rrms_ptr}}, {{rows_expr}}, {{c}}, {{eps}}f, stream
{{indent}});
"""
)


def _C(func_attrs):
    return func_attrs["inputs"][0]._attrs["shape"][-1]._attrs["values"][0]


@registry.reg("cuda.rms_quantize_mxfp8.gen_function")
def gen_function(func_attrs):
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.rms_quantize_mxfp8.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.rms_quantize_mxfp8.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xq, sfa, rrms = func_attrs["outputs"]
    rows_expr = " * ".join(d._attrs["name"] for d in x._attrs["shape"][:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent, func_name=func_attrs["name"],
        x_ptr=x._attrs["name"], xq_ptr=xq._attrs["name"], sfa_ptr=sfa._attrs["name"],
        rrms_ptr=rrms._attrs["name"], rows_expr=rows_expr, c=_C(func_attrs), eps=func_attrs["eps"],
    )
