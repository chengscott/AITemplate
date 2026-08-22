#  Backend codegen for the native `rmsnorm` op: a single-pass, one-warp-per-row fused
#  RMSNorm kernel (Z = RMSNorm(X) * gamma [+ relu]). fp32 accumulation matches the reference.
#  fp8_out variant: also quantizes the output to e4m3 per-row IN-KERNEL (holds z in a
#  register, warp-reduces its absmax, writes e4m3 + f32 scale), so a downstream fp8 gemm
#  needs no separate quantize kernel. Requires C <= 256 (VECS<=32 -> 1 uint4 chunk per lane).
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
{% if fp8_out %}#include <cuda_fp8.h>{% endif %}

namespace {
__global__ void {{func_name}}_kernel(const __half* __restrict__ x,
                                     const __half* __restrict__ gamma,
{% if fp8_out %}                                     __nv_fp8_e4m3* __restrict__ xq,
                                     float* __restrict__ scale,
{% else %}                                     __half* __restrict__ z,
{% endif %}                                     long long rows, int C, float eps) {
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const int VECS = C >> 3;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * (long long)C);
  const uint4* gr = reinterpret_cast<const uint4*>(gamma);

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

{% if fp8_out %}
  // compute z into a register uint4, warp-reduce its absmax, quantize -> e4m3 (VECS<=32).
  uint4 zq; float amax = 0.f;
  const bool active = (lane < VECS);
  if (active) {
    uint4 q = xr[lane];
    uint4 g = gr[lane];
    __half2* h = reinterpret_cast<__half2*>(&q);
    const __half2* gh = reinterpret_cast<const __half2*>(&g);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f = __half22float2(h[i]);
      float2 gf = __half22float2(gh[i]);
      float a = f.x * rrms * gf.x;
      float b = f.y * rrms * gf.y;
      {{relu_stmt}}
      amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
      h[i] = __float22half2_rn(make_float2(a, b));
    }
    zq = q;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float sc = amax * (1.0f / 448.0f);
  const float inv = sc > 0.f ? (1.0f / sc) : 0.f;
  if (lane == 0) scale[row] = sc > 0.f ? sc : 1.0f;
  if (active) {
    const __half2* zh = reinterpret_cast<const __half2*>(&zq);
    uint2 out;
    unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f = __half22float2(zh[i]);
      f.x *= inv; f.y *= inv;
      os[i] = __nv_fp8x2_e4m3(f).__x;
    }
    reinterpret_cast<uint2*>(xq + row * (long long)C)[lane] = out;
  }
{% else %}
  uint4* zr = reinterpret_cast<uint4*>(z + row * (long long)C);
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
{% endif %}
}
}  // namespace

void {{func_name}}(const void* x_ptr, const void* gamma_ptr,
{% if fp8_out %}                   void* xq_ptr, void* scale_ptr,
{% else %}                   void* z_ptr,
{% endif %}                   int64_t rows, int64_t C, float eps, cudaStream_t stream) {
  constexpr int BLK = 128;  // 4 warps/CTA, one row per warp
  const unsigned int grid = (unsigned int)((rows + (BLK >> 5) - 1) / (BLK >> 5));
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const __half*>(x_ptr),
      reinterpret_cast<const __half*>(gamma_ptr),
{% if fp8_out %}      reinterpret_cast<__nv_fp8_e4m3*>(xq_ptr),
      reinterpret_cast<float*>(scale_ptr),
{% else %}      reinterpret_cast<__half*>(z_ptr),
{% endif %}      (long long)rows, (int)C, eps);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, const void*, void*, {% if fp8_out %}void*, {% endif %}"
    "int64_t, int64_t, float, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{gamma_ptr}}, {{z_ptr}},{% if scale_ptr %} {{scale_ptr}},{% endif %}
{{indent}}    {{rows_expr}}, {{c}}, {{eps}}f, stream
{{indent}});
"""
)


@registry.reg("cuda.rmsnorm.gen_function")
def rmsnorm_gen_function(func_attrs):
    relu_stmt = "a = fmaxf(a, 0.f); b = fmaxf(b, 0.f);" if func_attrs.get("relu") else ""
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        relu_stmt=relu_stmt,
        fp8_out=func_attrs.get("fp8_out", False),
    )


@registry.reg("cuda.rmsnorm.func_decl")
def rmsnorm_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"], fp8_out=func_attrs.get("fp8_out", False)
    )


@registry.reg("cuda.rmsnorm.func_call")
def rmsnorm_gen_function_call(func_attrs, indent="  "):
    x, gamma = func_attrs["inputs"]
    fp8_out = func_attrs.get("fp8_out", False)
    z = func_attrs["outputs"][0]
    scale_ptr = func_attrs["outputs"][1]._attrs["name"] if fp8_out else None
    x_shape = x._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    c = x_shape[-1]._attrs["values"][0]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        gamma_ptr=gamma._attrs["name"],
        z_ptr=z._attrs["name"],
        scale_ptr=scale_ptr,
        rows_expr=rows_expr,
        c=c,
        eps=func_attrs["eps"],
    )
