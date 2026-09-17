#  Backend codegen for `im2col_fp8`: one warp per output position (b,oh,ow) gathers the
#  kh*kw*Cin (zero-padded) window, computes the fp32 row absmax (warp reduce), writes the
#  per-row f32 scale = amax/448, then re-gathers writing xq = to_e4m3(window / scale).
#  Column order is (ky,kx,c) to match the NHWC conv weight [Cout,kh,kw,Cin] reshaped to
#  [Cout, kh*kw*Cin]. See compiler/ops/gemm_universal/im2col_fp8.py.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {
constexpr float kE4M3Max = 448.0f;

// One WARP per output row r = (b*OH + oh)*OW + ow. K = kh*kw*Cin, col = (ky*kw+kx)*Cin+c.
__global__ void {{func_name}}_kernel(const __half* __restrict__ x,
                                     __nv_fp8_e4m3* __restrict__ xq,
                                     float* __restrict__ scale, long long rows) {
  const int H = {{H}}, W = {{W}}, Cin = {{Cin}};
  const int OH = {{OH}}, OW = {{OW}}, KH = {{kh}}, KW = {{kw}};
  const int STRIDE = {{stride}}, PAD = {{pad}}, K = KH * KW * Cin;
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const int ow = (int)(row % OW);
  const int oh = (int)((row / OW) % OH);
  const long long b = row / ((long long)OH * OW);
  const long long xb = b * H * W * Cin;

  float amax = 0.f;
  for (int idx = lane; idx < K; idx += 32) {
    const int c = idx % Cin;
    const int kk = idx / Cin;         // ky*KW + kx
    const int ih = oh * STRIDE - PAD + kk / KW;
    const int iw = ow * STRIDE - PAD + kk % KW;
    if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
      amax = fmaxf(amax, fabsf(__half2float(x[xb + ((long long)ih * W + iw) * Cin + c])));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float sc = amax * (1.0f / kE4M3Max);
  const float inv = sc > 0.f ? (1.0f / sc) : 0.f;
  if (lane == 0) scale[row] = sc > 0.f ? sc : 1.0f;

  __nv_fp8_e4m3* outr = xq + row * (long long)K;
  for (int idx = lane; idx < K; idx += 32) {
    const int c = idx % Cin;
    const int kk = idx / Cin;
    const int ih = oh * STRIDE - PAD + kk / KW;
    const int iw = ow * STRIDE - PAD + kk % KW;
    float v = 0.f;
    if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
      v = __half2float(x[xb + ((long long)ih * W + iw) * Cin + c]);
    outr[idx] = __nv_fp8_e4m3(v * inv);
  }
}
}  // namespace

// x [B,H,W,Cin] f16 -> xq [rows,K] e4m3 + scale [rows] f32, rows = B*OH*OW. No workspace.
void {{func_name}}(const void* x_ptr, void* xq_ptr, void* scale_ptr, int64_t rows,
                   cudaStream_t stream) {
  constexpr int BLK = 128;  // 4 warps/CTA, one output row per warp
  const unsigned int grid = (unsigned int)((rows + (BLK >> 5) - 1) / (BLK >> 5));
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const __half*>(x_ptr),
      reinterpret_cast<__nv_fp8_e4m3*>(xq_ptr),
      reinterpret_cast<float*>(scale_ptr), (long long)rows);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, void*, void*, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{xq_ptr}}, {{scale_ptr}}, {{rows_expr}}, stream
{{indent}});
"""
)


@registry.reg("cuda.im2col_fp8.gen_function")
def gen_function(func_attrs):
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        H=func_attrs["H"],
        W=func_attrs["W"],
        Cin=func_attrs["Cin"],
        OH=func_attrs["OH"],
        OW=func_attrs["OW"],
        kh=func_attrs["kh"],
        kw=func_attrs["kw"],
        stride=func_attrs["stride"],
        pad=func_attrs["pad"],
    )


@registry.reg("cuda.im2col_fp8.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.im2col_fp8.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xq = func_attrs["outputs"][0]
    scale = func_attrs["outputs"][1]
    B = x._attrs["shape"][0]
    rows_expr = "({}) * {}".format(
        B._attrs["name"], func_attrs["OH"] * func_attrs["OW"]
    )
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        xq_ptr=xq._attrs["name"],
        scale_ptr=scale._attrs["name"],
        rows_expr=rows_expr,
    )
