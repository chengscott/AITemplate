#  Backend codegen for the SM100 MXFP8 (block-scaled e4m3) gemm with a FUSED activation quantize.
#  The emitted function (1) quantizes the f16 activation A [M,K] to e4m3 + per-32-element ue8m0
#  block scales (SFA, written in the swizzled cuBLAS layout) into static device workspaces, then
#  (2) runs the tcgen05 block-scaled UMMA against the e4m3 weight B [N,K] + its baked swizzled
#  ue8m0 scales (SFB). Block scales apply inside the MMA -> the epilogue is a plain
#  LinearCombination (alpha=1, beta=residual) -> f16 out. N,K baked; M runtime. SM100 only.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace {{func_name}}_ns {
using namespace cute;
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementOut = cutlass::half_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using TileShapeMNK    = {{tile}};
using ClusterShapeMNK = {{cluster}};
constexpr int AlignA = 16, AlignB = 16;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, TileShapeMNK, ClusterShapeMNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc,
    ElementOut, LayoutC, AlignC, ElementOut, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAcc, TileShapeMNK, ClusterShapeMNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using DataA   = typename ElementA::DataType;         // e4m3
using ScaleT  = typename ElementA::ScaleFactorType;  // ue8m0
using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Quantize activation -> aq e4m3 [M,K] + sfa (ue8m0, swizzled). One warp/row, each lane owns
// 32-elem K-blocks (nkb). e8m0 = round-up-pow2(amax/448). norm=='rmsnorm': fold RMSNorm+gamma
// (+relu) into the quantize (one read of the raw input; holds the row in NCHUNK uint4/lane),
// eliminating the separate norm kernel + its f16 output round-trip. norm=='none': quantize a
// directly. (This keeps sfa a STATIC workspace inside the gemm op -- no dynamic SF graph tensor.)
__global__ void {{func_name}}_quant(const __half* __restrict__ a,
{% if norm == 'rmsnorm' %}                                    const __half* __restrict__ gamma, float eps,
{% endif %}{% if norm == 'swiglu' %}                                    const __half* __restrict__ rrms,
{% endif %}{% if norm == 'rms_out' %}                                    __half* __restrict__ rrms_out, float eps,
{% endif %}                                    unsigned char* __restrict__ aq,
                                    unsigned char* __restrict__ sfa, long long rows, int nkb, int ntx) {
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const long long K = (long long)nkb * 32;
  const uint4* ar = reinterpret_cast<const uint4*>(a + row * K);   // 8 half / uint4
  uint2* aqr = reinterpret_cast<uint2*>(aq + row * K);             // 8 e4m3 / uint2
  const long long iy = row & 127;
{% if norm == 'rmsnorm' %}
  const uint4* gr = reinterpret_cast<const uint4*>(gamma);
  uint4 xbuf[{{NCHUNK}}][4]; float ss = 0.f;
  int bi = 0;
  for (int kb = lane; kb < nkb; kb += 32, bi++) {
#pragma unroll
    for (int j = 0; j < 4; j++) { uint4 q = ar[kb * 4 + j]; xbuf[bi][j] = q; const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); ss += f.x * f.x + f.y * f.y; } }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float rrms = rsqrtf(ss / (float)K + eps);
  bi = 0;
  for (int kb = lane; kb < nkb; kb += 32, bi++) {
    float amax = 0.f; uint4 nb[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const __half2* h = reinterpret_cast<const __half2*>(&xbuf[bi][j]);
      uint4 g = gr[kb * 4 + j]; const __half2* gh = reinterpret_cast<const __half2*>(&g);
      __half2* nh = reinterpret_cast<__half2*>(&nb[j]);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); float2 gf = __half22float2(gh[i]);
        float av = f.x * rrms * gf.x, bv = f.y * rrms * gf.y; {{relu_stmt}}
        amax = fmaxf(amax, fmaxf(fabsf(av), fabsf(bv))); nh[i] = __float22half2_rn(make_float2(av, bv)); }
    }
    int b = amax > 0.f ? ((int)ceilf(__log2f(amax * (1.0f / 448.0f))) + 127) : 0; b = b < 0 ? 0 : (b > 254 ? 254 : b);
    float inv = amax > 0.f ? exp2f((float)(127 - b)) : 0.f;
    sfa[(row >> 7) * (long long)ntx * 512 + (kb >> 2) * 512 + (iy % 32) * 16 + (iy >> 5) * 4 + (kb & 3)] = (unsigned char)b;
#pragma unroll
    for (int j = 0; j < 4; j++) { const __half2* zh = reinterpret_cast<const __half2*>(&nb[j]);
      uint2 out; unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(zh[i]); f.x *= inv; f.y *= inv; os[i] = __nv_fp8x2_e4m3(f).__x; }
      aqr[kb * 4 + j] = out; }
  }
{% elif norm == 'rms_out' %}
  // qkv/fc1 rmsnorm-prologue: quantize RAW x (gamma is folded into the weight) AND emit per-row
  // rrms = rsqrt(mean(x^2)+eps) as a 2nd output (folds ops.rms_reduce -> no separate read of x).
  // unpack_rope / fc2-swiglu apply rrms downstream. One read of x (held in xbuf for the quantize).
  uint4 xbuf[{{NCHUNK}}][4]; float ss = 0.f; int bi = 0;
  for (int kb = lane; kb < nkb; kb += 32, bi++) {
#pragma unroll
    for (int j = 0; j < 4; j++) { uint4 q = ar[kb * 4 + j]; xbuf[bi][j] = q; const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); ss += f.x * f.x + f.y * f.y; } }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float rrms = rsqrtf(ss / (float)K + eps);
  if (lane == 0) rrms_out[row] = __float2half(rrms);
  bi = 0;
  for (int kb = lane; kb < nkb; kb += 32, bi++) {
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) { const __half2* h = reinterpret_cast<const __half2*>(&xbuf[bi][j]);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y))); } }
    int b = amax > 0.f ? ((int)ceilf(__log2f(amax * (1.0f / 448.0f))) + 127) : 0; b = b < 0 ? 0 : (b > 254 ? 254 : b);
    float inv = amax > 0.f ? exp2f((float)(127 - b)) : 0.f;
    sfa[(row >> 7) * (long long)ntx * 512 + (kb >> 2) * 512 + (iy % 32) * 16 + (iy >> 5) * 4 + (kb & 3)] = (unsigned char)b;
#pragma unroll
    for (int j = 0; j < 4; j++) { const __half2* h = reinterpret_cast<const __half2*>(&xbuf[bi][j]);
      uint2 out; unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); f.x *= inv; f.y *= inv; os[i] = __nv_fp8x2_e4m3(f).__x; }
      aqr[kb * 4 + j] = out; }
  }
{% elif norm == 'swiglu' %}
  // fc2 input = SwiGLU(fc1): z[j] = silu(rf*gate[j]) * (rf*up[j]), rf = rrms[row] (rmsnorm-prologue).
  // fc1 [row, 2K]: gate = [0..K), up = [K..2K). Read gate+up blocks, apply silu*mul, per-32-block quant.
  const uint4* fr = reinterpret_cast<const uint4*>(a + row * (2 * K));  // fc1 row (stride 2K halfs)
  const int upv = (int)(K >> 3);                                       // uint4 offset to up half (K/8)
  const float rf = __half2float(rrms[row]);
  for (int kb = lane; kb < nkb; kb += 32) {
    float amax = 0.f; uint4 zb[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint4 gq = fr[kb * 4 + j]; uint4 uq = fr[upv + kb * 4 + j];
      const __half2* gh = reinterpret_cast<const __half2*>(&gq);
      const __half2* uh = reinterpret_cast<const __half2*>(&uq);
      __half2* zh = reinterpret_cast<__half2*>(&zb[j]);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 gf = __half22float2(gh[i]); float2 uf = __half22float2(uh[i]);
        float gx = gf.x * rf, gy = gf.y * rf;
        float av = (gx / (1.f + __expf(-gx))) * (uf.x * rf);
        float bv = (gy / (1.f + __expf(-gy))) * (uf.y * rf);
        amax = fmaxf(amax, fmaxf(fabsf(av), fabsf(bv))); zh[i] = __float22half2_rn(make_float2(av, bv)); }
    }
    int b = amax > 0.f ? ((int)ceilf(__log2f(amax * (1.0f / 448.0f))) + 127) : 0; b = b < 0 ? 0 : (b > 254 ? 254 : b);
    float inv = amax > 0.f ? exp2f((float)(127 - b)) : 0.f;
    sfa[(row >> 7) * (long long)ntx * 512 + (kb >> 2) * 512 + (iy % 32) * 16 + (iy >> 5) * 4 + (kb & 3)] = (unsigned char)b;
#pragma unroll
    for (int j = 0; j < 4; j++) { const __half2* zh = reinterpret_cast<const __half2*>(&zb[j]);
      uint2 out; unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(zh[i]); f.x *= inv; f.y *= inv; os[i] = __nv_fp8x2_e4m3(f).__x; }
      aqr[kb * 4 + j] = out; }
  }
{% else %}
  for (int kb = lane; kb < nkb; kb += 32) {
    uint4 buf[4]; float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) { uint4 q = ar[kb * 4 + j]; buf[j] = q; const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y))); } }
    int b = amax > 0.f ? ((int)ceilf(__log2f(amax * (1.0f / 448.0f))) + 127) : 0; b = b < 0 ? 0 : (b > 254 ? 254 : b);
    float inv = amax > 0.f ? exp2f((float)(127 - b)) : 0.f;
    sfa[(row >> 7) * (long long)ntx * 512 + (kb >> 2) * 512 + (iy % 32) * 16 + (iy >> 5) * 4 + (kb & 3)] = (unsigned char)b;
#pragma unroll
    for (int j = 0; j < 4; j++) { const __half2* h = reinterpret_cast<const __half2*>(&buf[j]);
      uint2 out; unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); f.x *= inv; f.y *= inv; os[i] = __nv_fp8x2_e4m3(f).__x; }
      aqr[kb * 4 + j] = out; }
  }
{% endif %}
}

#ifndef AIT_CUTLASS_CHECK
#define AIT_CUTLASS_CHECK(status)                                              \\
  { cutlass::Status s = (status);                                             \\
    if (s != cutlass::Status::kSuccess)                                        \\
      throw std::runtime_error(std::string("CUTLASS mxfp8 gemm error: ") +     \\
                               cutlassGetStatusString(s)); }
#endif
}  // namespace {{func_name}}_ns

// A [M,{{K}}] f16 (quantized internally),
// B [{{N}},{{K}}] e4m3 + b_scale (ue8m0 swizzled SFB, baked) -> D [M,{{N}}] f16.
void {{func_name}}(const void* a_ptr, {% if norm == 'rmsnorm' %}const void* gamma_ptr, {% endif %}{% if norm == 'swiglu' %}const void* rrms_ptr, {% endif %}const void* b_ptr, const void* bscale_ptr,
                   const void* residual_ptr, void* out_ptr, {% if norm == 'rms_out' %}void* rrms_out_ptr, {% endif %}int64_t M, cudaStream_t stream) {
  using namespace {{func_name}}_ns;
  const int N = {{N}}, K = {{K}}, nkb = K / 32, ntx = (nkb + 3) / 4; (void)nkb; (void)ntx;
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(M), K, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {static_cast<int>(M), N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {static_cast<int>(M), N, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(static_cast<int>(M), N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(static_cast<int>(M), N, K, 1));
  // static workspaces: quantized activation (e4m3) + its swizzled SFA (ue8m0)
  static DataA* s_aq = nullptr; static size_t s_aq_m = 0;
  static unsigned char* s_sfa = nullptr; static size_t s_sfa_sz = 0;
  if ((size_t)M > s_aq_m) { if (s_aq) cudaFree(s_aq); cudaMalloc(&s_aq, sizeof(DataA) * (size_t)M * K); s_aq_m = (size_t)M; }
  size_t sfa_need = (size_t)cute::size(cute::filter_zeros(layout_SFA));
  if (sfa_need > s_sfa_sz) { if (s_sfa) cudaFree(s_sfa); cudaMalloc(&s_sfa, sfa_need); s_sfa_sz = sfa_need; }
  {
    const int BLK = 128; unsigned int g = (unsigned int)((M + (BLK >> 5) - 1) / (BLK >> 5));
    {{func_name}}_quant<<<g, BLK, 0, stream>>>(reinterpret_cast<const __half*>(a_ptr),
{% if norm == 'rmsnorm' %}                                               reinterpret_cast<const __half*>(gamma_ptr), {{eps}}f,
{% endif %}{% if norm == 'swiglu' %}                                               reinterpret_cast<const __half*>(rrms_ptr),
{% endif %}{% if norm == 'rms_out' %}                                               reinterpret_cast<__half*>(rrms_out_ptr), {{eps}}f,
{% endif %}                                               reinterpret_cast<unsigned char*>(s_aq), s_sfa,
                                               (long long)M, nkb, ntx);
  }
  const DataA* a_e4m3 = s_aq;
  const ScaleT* a_sf = reinterpret_cast<const ScaleT*>(s_sfa);
  cutlass::KernelHardwareInfo hw_info;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {static_cast<int>(M), N, K, 1},
      {a_e4m3, stride_A, reinterpret_cast<const DataA*>(b_ptr), stride_B,
       a_sf, layout_SFA, reinterpret_cast<const ScaleT*>(bscale_ptr), layout_SFB},
      {{'{'}}{1.0f, {{ '1.0f' if has_residual else '0.0f' }}},
       reinterpret_cast<const ElementOut*>({{ 'residual_ptr' if has_residual else 'out_ptr' }}), stride_C,
       reinterpret_cast<ElementOut*>(out_ptr), stride_D{{'}'}},
      hw_info};
  Gemm gemm_op;
  static uint8_t* s_ws = nullptr; static size_t s_ws_sz = 0;
  size_t need = Gemm::get_workspace_size(arguments);
  if (need > s_ws_sz) { if (s_ws) cudaFree(s_ws); cudaMalloc(&s_ws, need); s_ws_sz = need; }
  AIT_CUTLASS_CHECK(gemm_op.can_implement(arguments));
  AIT_CUTLASS_CHECK(gemm_op.initialize(arguments, s_ws, stream));
  AIT_CUTLASS_CHECK(gemm_op.run(stream));
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, "
    "{% if norm == 'rmsnorm' %}const void*, {% endif %}{% if norm == 'swiglu' %}const void*, {% endif %}const void*, const void*, "
    "const void*, void*, {% if norm == 'rms_out' %}void*, {% endif %}int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}}, {% if gamma_ptr %}{{gamma_ptr}}, {% endif %}{% if rrms_ptr %}{{rrms_ptr}}, {% endif %}{{b_ptr}}, {{bscale_ptr}}, {{residual_ptr}}, {{out_ptr}}, {% if rrms_out_ptr %}{{rrms_out_ptr}}, {% endif %}{{m_expr}}, stream
{{indent}});
"""
)


def _tile_cluster():
    import os

    if os.environ.get("AIT_FP8_GEMM_2SM", "0") == "1":
        return "Shape<_256, _128, _128>", "Shape<_2, _1, _1>"
    return "Shape<_128, _128, _128>", "Shape<_1, _1, _1>"


@registry.reg("cuda.gemm_rcr_mxfp8.gen_function")
def gen_function(func_attrs):
    tile, cluster = _tile_cluster()
    K = func_attrs["K"]
    nkb = K // 32
    relu_stmt = "av = fmaxf(av, 0.f); bv = fmaxf(bv, 0.f);" if func_attrs.get("relu") else ""
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"], N=func_attrs["N"], K=K,
        has_residual=func_attrs.get("has_residual", False),
        norm=func_attrs.get("norm", "none"), relu_stmt=relu_stmt,
        NCHUNK=(nkb + 31) // 32, eps=func_attrs.get("eps", 1e-6),
        tile=tile, cluster=cluster,
    )


@registry.reg("cuda.gemm_rcr_mxfp8.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        norm=func_attrs.get("norm", "none"),
    )


@registry.reg("cuda.gemm_rcr_mxfp8.func_call")
def gen_function_call(func_attrs, indent="  "):
    # inputs: [a, b, b_scale] + [gamma? (rmsnorm)] + [rrms? (swiglu)] + [residual?]
    ins = func_attrs["inputs"]
    a, b, bscale = ins[0], ins[1], ins[2]
    idx = 3
    gamma_ptr = None
    if func_attrs.get("norm", "none") == "rmsnorm":
        gamma_ptr = ins[idx]._attrs["name"]; idx += 1
    rrms_ptr = None
    if func_attrs.get("norm", "none") == "swiglu":
        rrms_ptr = ins[idx]._attrs["name"]; idx += 1
    residual_ptr = ins[idx]._attrs["name"] if func_attrs.get("has_residual", False) else "nullptr"
    out = func_attrs["outputs"][0]
    rrms_out_ptr = None
    if func_attrs.get("norm", "none") == "rms_out":
        rrms_out_ptr = func_attrs["outputs"][1]._attrs["name"]
    m_expr = " * ".join(d._attrs["name"] for d in a._attrs["shape"][:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent, func_name=func_attrs["name"],
        a_ptr=a._attrs["name"], gamma_ptr=gamma_ptr, rrms_ptr=rrms_ptr,
        b_ptr=b._attrs["name"], bscale_ptr=bscale._attrs["name"],
        residual_ptr=residual_ptr, out_ptr=out._attrs["name"], rrms_out_ptr=rrms_out_ptr, m_expr=m_expr,
    )
