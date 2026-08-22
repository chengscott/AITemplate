#  Backend for `quantize_to_fp8_amax`: single pass, no reduction -- scale = amax[0]/448 was
#  produced by the upstream conv's epilogue (conv2d_fp8 emit_amax). One kernel: divide + cast.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {
constexpr float kE4M3Max = 448.0f;
// Vectorized: 8 halfs per thread (uint4 load) -> 8 e4m3 (uint2 store) via packed
// __nv_fp8x2_e4m3. n is a multiple of 8 (last dim C % 8 == 0). ~2x the scalar bandwidth.
__global__ void {{func_name}}_kernel(const uint4* __restrict__ x8,
                                     uint2* __restrict__ xq8,
                                     const float* __restrict__ amax,
                                     float* __restrict__ scale, long long n8) {
  const float sc = amax[0] * (1.0f / kE4M3Max);
  const float inv = sc > 0.f ? (1.0f / sc) : 0.f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n8;
       i += (long long)gridDim.x * blockDim.x) {
    uint4 q = x8[i];
    const __half2* h = reinterpret_cast<const __half2*>(&q);
    uint2 out;
    unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __half22float2(h[j]);
      f.x *= inv; f.y *= inv;
      os[j] = __nv_fp8x2_e4m3(f).__x;
    }
    xq8[i] = out;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) scale[0] = sc > 0.f ? sc : 1.0f;
}
}  // namespace

void {{func_name}}(const void* x_ptr, const void* amax_ptr, void* xq_ptr, void* scale_ptr,
                   int64_t n, cudaStream_t stream) {
  constexpr int BLK = 256;
  const long long n8 = n >> 3;
  unsigned int grid = (unsigned int)((n8 + BLK - 1) / BLK);
  if (grid > 4096u) grid = 4096u;
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr),
      reinterpret_cast<uint2*>(xq_ptr),
      reinterpret_cast<const float*>(amax_ptr),
      reinterpret_cast<float*>(scale_ptr), n8);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, const void*, void*, void*, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{amax_ptr}}, {{xq_ptr}}, {{scale_ptr}}, {{n_expr}}, stream
{{indent}});
"""
)


@registry.reg("cuda.quantize_to_fp8_amax.gen_function")
def gen_function(func_attrs):
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.quantize_to_fp8_amax.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.quantize_to_fp8_amax.func_call")
def gen_function_call(func_attrs, indent="  "):
    x, amax = func_attrs["inputs"][0], func_attrs["inputs"][1]
    xq, scale = func_attrs["outputs"][0], func_attrs["outputs"][1]
    n_expr = " * ".join(d._attrs["name"] for d in x._attrs["shape"]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent, func_name=func_attrs["name"],
        x_ptr=x._attrs["name"], amax_ptr=amax._attrs["name"],
        xq_ptr=xq._attrs["name"], scale_ptr=scale._attrs["name"], n_expr=n_expr,
    )
