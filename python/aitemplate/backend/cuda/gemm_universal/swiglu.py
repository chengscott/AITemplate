#  Backend codegen for the `swiglu` op: Z = silu(X[..., :ffn]) * X[..., ffn:], reading the
#  packed fc1 output [..., 2*ffn] directly (no split). Ports TE's swiglu activation
#  (transformer_engine/common/activation/swiglu.cu -> tex.swiglu) as inline AIT codegen:
#  one elementwise kernel, and by consuming the packed input it removes the split op the
#  generic elementwise path needs. fp32 math, fp16 in/out.
#
#  rmsnorm-prologue: when an rrms [rows,1] input is given, each row's gate/up is multiplied
#  by rrms[row] here (the fc1 gemm ran on RAW x with gamma folded into its weight), so the
#  separate RMSNorm kernel's [rows,dim] write of the normalized activation is eliminated.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
{% if fp8_out %}#include <cuda_fp8.h>
{% endif %}
namespace {
// Z[r,j] = silu(rf*X[r,j]) * (rf*X[r, ffn+j]), rf = rrms[r] (1 if no prologue). Vectorized:
// one thread per 8 columns (uint4 = 8 halfs) of gate and up -> one uint4 store.
__global__ void {{func_name}}_kernel(const __half* __restrict__ x, __half* __restrict__ z,
{% if has_rrms %}                                     const __half* __restrict__ rrms,
{% endif %}                                     long long rows, int ffn) {
  const int ffn8 = ffn >> 3;
  const long long total = rows * (long long)ffn8;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long long r = i / ffn8;
  const int c8 = (int)(i - r * ffn8);
{% if has_rrms %}  const float rf = __half2float(rrms[r]);
{% else %}  const float rf = 1.0f;
{% endif %}  const __half* xr = x + r * (2LL * ffn);
  uint4 gq = reinterpret_cast<const uint4*>(xr)[c8];        // gate cols [8*c8 .. +7]
  uint4 uq = reinterpret_cast<const uint4*>(xr + ffn)[c8];  // up cols
  __half2* gh = reinterpret_cast<__half2*>(&gq);
  const __half2* uh = reinterpret_cast<const __half2*>(&uq);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float2 gf = __half22float2(gh[k]);
    float2 uf = __half22float2(uh[k]);
    float gx = gf.x * rf, gy = gf.y * rf;
    float a = (gx / (1.f + __expf(-gx))) * (uf.x * rf);
    float b = (gy / (1.f + __expf(-gy))) * (uf.y * rf);
    gh[k] = __float22half2_rn(make_float2(a, b));
  }
  reinterpret_cast<uint4*>(z + r * (long long)ffn)[c8] = gq;
}
// scalar fallback for ffn not a multiple of 8.
__global__ void {{func_name}}_kernel_scalar(const __half* __restrict__ x, __half* __restrict__ z,
{% if has_rrms %}                                            const __half* __restrict__ rrms,
{% endif %}                                            long long rows, int ffn) {
  const long long total = rows * (long long)ffn;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long long r = i / ffn;
  const int j = (int)(i - r * ffn);
{% if has_rrms %}  const float rf = __half2float(rrms[r]);
{% else %}  const float rf = 1.0f;
{% endif %}  const __half* xr = x + r * (2LL * ffn);
  const float g = __half2float(xr[j]) * rf;
  const float u = __half2float(xr[ffn + j]) * rf;
  z[i] = __float2half((g / (1.f + __expf(-g))) * u);
}
{% if fp8_out %}
// fp8-fused producer: WARP per row. Compute z=silu*up (held in registers), warp-reduce the
// row's absmax, quantize to e4m3 per-row -> zq [rows,ffn] e4m3 + scale [rows]. Feeds a
// downstream fp8 gemm (fc2) with no separate quantize pass. ffn % 8 == 0 required.
__global__ void {{func_name}}_fp8_kernel(const __half* __restrict__ x, uint2* __restrict__ zq,
                                         float* __restrict__ scale,
{% if has_rrms %}                                         const __half* __restrict__ rrms,
{% endif %}                                         long long rows, int ffn) {
  const int ffn8 = ffn >> 3;                 // uint4 chunks per row (VECS)
  const int lane = threadIdx.x & 31;
  const long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) return;
{% if has_rrms %}  const float rf = __half2float(rrms[r]);
{% else %}  const float rf = 1.0f;
{% endif %}  const uint4* gr = reinterpret_cast<const uint4*>(x + r * (2LL * ffn));
  const uint4* ur = reinterpret_cast<const uint4*>(x + r * (2LL * ffn) + ffn);
  uint4 zreg[{{NCHUNK}}];                     // computed z chunks, held for the quantize pass
  float amax = 0.f;
#pragma unroll
  for (int c = 0; c < {{NCHUNK}}; c++) {
    const int v = lane + c * 32;
    if (v < ffn8) {
      uint4 gq = gr[v];
      uint4 uq = ur[v];
      __half2* gh = reinterpret_cast<__half2*>(&gq);
      const __half2* uh = reinterpret_cast<const __half2*>(&uq);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float2 gf = __half22float2(gh[k]);
        float2 uf = __half22float2(uh[k]);
        float gx = gf.x * rf, gy = gf.y * rf;
        float a = (gx / (1.f + __expf(-gx))) * (uf.x * rf);
        float b = (gy / (1.f + __expf(-gy))) * (uf.y * rf);
        gh[k] = __float22half2_rn(make_float2(a, b));
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
      }
      zreg[c] = gq;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float sc = amax * (1.0f / 448.0f);
  const float inv = sc > 0.f ? (1.0f / sc) : 0.f;
  if (lane == 0) scale[r] = sc > 0.f ? sc : 1.0f;
  uint2* outr = zq + r * (long long)ffn8;
#pragma unroll
  for (int c = 0; c < {{NCHUNK}}; c++) {
    const int v = lane + c * 32;
    if (v < ffn8) {
      __half2* gh = reinterpret_cast<__half2*>(&zreg[c]);
      uint2 out;
      unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float2 f = __half22float2(gh[k]);
        f.x *= inv; f.y *= inv;
        os[k] = __nv_fp8x2_e4m3(f).__x;
      }
      outr[v] = out;
    }
  }
}
{% endif %}}  // namespace

{% if fp8_out %}
// Z[..., ffn] = SwiGLU(X) quantized to e4m3 per-row + scale [rows]. No workspace.
void {{func_name}}(const void* x_ptr, void* zq_ptr, void* scale_ptr,
                   {% if has_rrms %}const void* rrms_ptr, {% endif %}int64_t rows,
                   int64_t ffn, cudaStream_t stream) {
  const int block = 128;                       // 4 warps/CTA, one warp per row
  const unsigned int grid = (unsigned int)((rows + (block >> 5) - 1) / (block >> 5));
  {{func_name}}_fp8_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<uint2*>(zq_ptr),
      reinterpret_cast<float*>(scale_ptr),
{% if has_rrms %}      reinterpret_cast<const __half*>(rrms_ptr),
{% endif %}      (long long)rows, (int)ffn);
}
{% else %}
// Z[..., ffn] = SwiGLU(X[..., 2*ffn]). No workspace.
void {{func_name}}(const void* x_ptr, void* z_ptr, {% if has_rrms %}const void* rrms_ptr, {% endif %}int64_t rows,
                   int64_t ffn, cudaStream_t stream) {
  const int block = 256;
  if ((ffn & 7) == 0) {
    const long long total = (long long)rows * (ffn >> 3);
    const unsigned int grid = (unsigned int)((total + block - 1) / block);
    {{func_name}}_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<__half*>(z_ptr),
{% if has_rrms %}        reinterpret_cast<const __half*>(rrms_ptr),
{% endif %}        (long long)rows, (int)ffn);
  } else {
    const long long total = (long long)rows * ffn;
    const unsigned int grid = (unsigned int)((total + block - 1) / block);
    {{func_name}}_kernel_scalar<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<__half*>(z_ptr),
{% if has_rrms %}        reinterpret_cast<const __half*>(rrms_ptr),
{% endif %}        (long long)rows, (int)ffn);
  }
}
{% endif %}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, void*, {% if fp8_out %}void*, {% endif %}{% if has_rrms %}const void*, {% endif %}int64_t, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{z_ptr}}, {% if fp8_out %}{{scale_ptr}}, {% endif %}{% if rrms %}{{rrms}}, {% endif %}{{rows_expr}}, {{ffn}}, stream
{{indent}});
"""
)


def _ffn(func_attrs):
    return func_attrs["inputs"][0]._attrs["shape"][-1]._attrs["values"][0] // 2


@registry.reg("cuda.swiglu.gen_function")
def swiglu_gen_function(func_attrs):
    ffn8 = _ffn(func_attrs) >> 3
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        has_rrms=func_attrs.get("has_rrms", False),
        fp8_out=func_attrs.get("fp8_out", False),
        NCHUNK=(ffn8 + 31) // 32,
    )


@registry.reg("cuda.swiglu.func_decl")
def swiglu_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        has_rrms=func_attrs.get("has_rrms", False),
        fp8_out=func_attrs.get("fp8_out", False),
    )


@registry.reg("cuda.swiglu.func_call")
def swiglu_gen_function_call(func_attrs, indent="  "):
    has_rrms = func_attrs.get("has_rrms", False)
    fp8_out = func_attrs.get("fp8_out", False)
    x = func_attrs["inputs"][0]
    rrms = func_attrs["inputs"][1]._attrs["name"] if has_rrms else None
    z = func_attrs["outputs"][0]
    scale = func_attrs["outputs"][1]._attrs["name"] if fp8_out else None
    x_shape = x._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    ffn = x_shape[-1]._attrs["values"][0] // 2
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        z_ptr=z._attrs["name"],
        scale_ptr=scale,
        fp8_out=fp8_out,
        rrms=rrms,
        rows_expr=rows_expr,
        ffn=ffn,
    )
