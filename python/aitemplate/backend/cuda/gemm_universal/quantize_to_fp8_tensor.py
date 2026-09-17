#  Backend codegen for `quantize_to_fp8_tensor`: two launches -- a grid-stride block
#  reduction (atomicMax over int bits, valid since |x|>=0) computes the global absmax into a
#  static scratch scalar, then a quantize pass writes xq = to_e4m3(x / (amax/448)) and the
#  f32 scale. See compiler/ops/gemm_universal/quantize_to_fp8_tensor.py.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {
constexpr float kE4M3Max = 448.0f;

__global__ void {{func_name}}_amax(const uint4* __restrict__ x8, float* __restrict__ amax,
                                   long long n8) {
  __shared__ float sm[32];
  float v = 0.f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n8;
       i += (long long)gridDim.x * blockDim.x) {
    uint4 q = x8[i];
    const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __half22float2(h[j]);
      v = fmaxf(v, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  // warp reduce
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) sm[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) atomicMax(reinterpret_cast<int*>(amax), __float_as_int(v));
  }
}

__global__ void {{func_name}}_quant(const uint4* __restrict__ x8,
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

// x [n] f16 -> xq [n] e4m3 + scale [1] f32 (global amax/448).
void {{func_name}}(const void* x_ptr, void* xq_ptr, void* scale_ptr, int64_t n,
                   cudaStream_t stream) {
  static float* s_amax = nullptr;
  if (!s_amax) cudaMalloc(&s_amax, sizeof(float));
  cudaMemsetAsync(s_amax, 0, sizeof(float), stream);
  constexpr int BLK = 256;
  const long long n8 = n >> 3;  // 8 halfs / uint4 (last dim C % 8 == 0)
  unsigned int grid = (unsigned int)((n8 + BLK - 1) / BLK);
  if (grid > 4096u) grid = 4096u;
  {{func_name}}_amax<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr), s_amax, n8);
  {{func_name}}_quant<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr),
      reinterpret_cast<uint2*>(xq_ptr), s_amax,
      reinterpret_cast<float*>(scale_ptr), n8);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, void*, void*, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{xq_ptr}}, {{scale_ptr}}, {{n_expr}}, stream
{{indent}});
"""
)


@registry.reg("cuda.quantize_to_fp8_tensor.gen_function")
def gen_function(func_attrs):
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.quantize_to_fp8_tensor.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.quantize_to_fp8_tensor.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xq = func_attrs["outputs"][0]
    scale = func_attrs["outputs"][1]
    n_expr = " * ".join(d._attrs["name"] for d in x._attrs["shape"]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        xq_ptr=xq._attrs["name"],
        scale_ptr=scale._attrs["name"],
        n_expr=n_expr,
    )
