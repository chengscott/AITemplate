#  Backend codegen for `quantize_to_fp8`: per-ROW (per-token) dynamic fp8 quantization.
#  A "group" of G lanes owns one row (128-bit uint4 loads, fp32 group-shuffle absmax, packed
#  __nv_fp8x2_e4m3 uint2 stores). G = min(32, VECS) so that for small C (e.g. C=128 -> VECS=16)
#  TWO rows share a warp (G=16, no idle lanes); for C>=256 it's one row per warp (G=32). C is
#  baked at codegen. See compiler/ops/gemm_universal/quantize_to_fp8.py.
import jinja2

from aitemplate.backend import registry


def _group(C):
    vecs = C >> 3
    g = 32
    while g > 1 and (g >> 1) >= vecs:
        g >>= 1
    return g  # largest power-of-2 <= VECS, capped at 32 (>=1)


FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {
constexpr float kE4M3Max = 448.0f;
// {{G}} lanes per row; {{RPW}} rows per warp (32/{{G}}). glane in [0,{{G}}) owns the row's
// uint4 chunks (stride {{G}}); group-shuffle absmax uses offsets < {{G}} (stay in the group).
__global__ void {{func_name}}_kernel(const uint4* __restrict__ x,
                                     uint2* __restrict__ xq,
                                     float* __restrict__ scale, long long rows, int VECS) {
  const int lane = threadIdx.x & 31;
  const int glane = lane % {{G}};
  const int grow = lane / {{G}};
  const long long row =
      ((long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * {{RPW}} + grow;
  // NOTE: do NOT early-return -- all 32 lanes must reach the group shuffle (partial warps at
  // odd `rows`, e.g. B=1). Inactive lanes contribute amax=0 and skip the memory ops.
  const bool active = row < rows;
  const uint4* xr = x + (active ? row : 0) * (long long)VECS;
  float amax = 0.f;
  if (active)
    for (int v = glane; v < VECS; v += {{G}}) {
      uint4 q = xr[v];
      const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __half22float2(h[i]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
#pragma unroll
  for (int o = {{G}} >> 1; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float sc = amax * (1.0f / kE4M3Max);
  const float inv = sc > 0.f ? (1.0f / sc) : 0.f;
  if (active && glane == 0) scale[row] = sc > 0.f ? sc : 1.0f;
  if (!active) return;
  uint2* outr = xq + row * (long long)VECS;
  for (int v = glane; v < VECS; v += {{G}}) {
    uint4 q = xr[v];
    const __half2* h = reinterpret_cast<const __half2*>(&q);
    uint2 out;
    unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f = __half22float2(h[i]);
      f.x *= inv; f.y *= inv;
      os[i] = __nv_fp8x2_e4m3(f).__x;
    }
    outr[v] = out;
  }
}
}  // namespace

// x [rows, C] f16 -> xq [rows, C] e4m3 + scale [rows] f32. C % 8 == 0. No workspace.
void {{func_name}}(const void* x_ptr, void* xq_ptr, void* scale_ptr, int64_t rows,
                   int64_t C, cudaStream_t stream) {
  constexpr int BLK = 128;                 // 4 warps/CTA
  constexpr int RPW = {{RPW}};             // rows per warp
  const long long rows_per_cta = (BLK >> 5) * RPW;
  const unsigned int grid = (unsigned int)((rows + rows_per_cta - 1) / rows_per_cta);
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr), reinterpret_cast<uint2*>(xq_ptr),
      reinterpret_cast<float*>(scale_ptr), (long long)rows, (int)(C >> 3));
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, void*, void*, int64_t, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{xq_ptr}}, {{scale_ptr}}, {{rows_expr}}, {{c}}, stream
{{indent}});
"""
)


def _C(func_attrs):
    return func_attrs["inputs"][0]._attrs["shape"][-1]._attrs["values"][0]


@registry.reg("cuda.quantize_to_fp8.gen_function")
def gen_function(func_attrs):
    g = _group(_C(func_attrs))
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"], G=g, RPW=32 // g)


@registry.reg("cuda.quantize_to_fp8.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.quantize_to_fp8.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xq = func_attrs["outputs"][0]
    scale = func_attrs["outputs"][1]
    x_shape = x._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        xq_ptr=xq._attrs["name"],
        scale_ptr=scale._attrs["name"],
        rows_expr=rows_expr,
        c=_C(func_attrs),
    )
