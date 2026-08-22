#  Backend codegen for the FUSED fp8 gemm (gemm_rcr_fp8_fused): CUTLASS 3x SM90 fp8 gemm
#  (RowMajor A, ColumnMajor B) whose epilogue folds dequant + per-N bias + residual:
#     D_f16 = alpha * (A@B^T) + beta*residual + bias[n],  alpha = scale_x*scale_w
#  via LinCombPerColBiasEltAct. Writes f16 directly (no f32 acc round-trip / dequant kernel).
#  N,K baked; M runtime.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace {{func_name}}_ns {
using namespace cute;

using ElementA       = cutlass::float_e4m3_t;
using ElementB       = cutlass::float_e4m3_t;
using ElementOut     = cutlass::half_t;
using ElementAcc     = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
// Tile chosen by N (swept on H200): N%256==0 -> 128x256 (fc1/up), N==384 -> 256x128 (qkv),
// N<=128 -> 128x128 (o/fc2/down). Cluster 1x1x1 (2x1 was not faster).
using TileShapeMNK    = {{tile}};
using ClusterShapeMNK = Shape<_1, _1, _1>;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;   // 16
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;   // 16
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value; // 8

// Per-ROW (per-token) scale: D = alpha[m]*acc + beta*C (+ per-row bias, unused=0).
using FusionOp = cutlass::epilogue::fusion::PerRowLinCombPerRowBiasEltAct<
    cutlass::epilogue::thread::Identity, ElementOut, ElementCompute, ElementOut,
    ElementOut, float>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShapeMNK, ClusterShapeMNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute,
    ElementOut, LayoutC, AlignC,
    ElementOut, LayoutC, AlignC,
    cutlass::epilogue::TmaWarpSpecializedCooperative, FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc, TileShapeMNK, ClusterShapeMNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#ifndef AIT_CUTLASS_CHECK
#define AIT_CUTLASS_CHECK(status)                                              \\
  {                                                                            \\
    cutlass::Status s = (status);                                             \\
    if (s != cutlass::Status::kSuccess)                                        \\
      throw std::runtime_error(std::string("CUTLASS gemm error: ") +          \\
                               cutlassGetStatusString(s));                     \\
  }
#endif

// per-row alpha[m] = scale_x[m] * scale_w[0]  (scale_x is the per-token quantize scale)
__global__ void alpha_kernel(const float* sx, const float* sw, float* o, long long M) {
  const float w = sw[0];
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < M;
       i += (long long)gridDim.x * blockDim.x)
    o[i] = sx[i] * w;
}
}  // namespace {{func_name}}_ns

// A [M,{{K}}] e4m3, B [{{N}},{{K}}] e4m3 -> D [M,{{N}}] f16 (fused per-row scale + residual)
void {{func_name}}(const void* a_ptr, const void* b_ptr, const void* scale_x_ptr,
                   const void* scale_w_ptr, const void* residual_ptr,
                   void* out_ptr, int64_t M, cudaStream_t stream) {
  using namespace {{func_name}}_ns;
  const int N = {{N}}, K = {{K}};
  static float* s_alpha = nullptr;
  static size_t s_alpha_m = 0;
  if ((size_t)M > s_alpha_m) {
    if (s_alpha) cudaFree(s_alpha);
    cudaMalloc(&s_alpha, sizeof(float) * (size_t)M);
    s_alpha_m = (size_t)M;
  }
  {
    unsigned int g = (unsigned int)((M + 255) / 256);
    if (g > 4096u) g = 4096u;
    alpha_kernel<<<g, 256, 0, stream>>>(
        reinterpret_cast<const float*>(scale_x_ptr),
        reinterpret_cast<const float*>(scale_w_ptr), s_alpha, (long long)M);
  }

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(M), K, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {static_cast<int>(M), N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {static_cast<int>(M), N, 1});

  // Default hw_info (sm_count=0) -> cutlass queries the SM count internally at run, same as
  // the conv path. (Calling KernelHardwareInfo::query_device_multiprocessor_count here pulls
  // in a fragile `_cudaGetDevice` symbol that fails to resolve when the .so is dlopen'd on a
  // fresh node -- see [[fp8-path]].)
  cutlass::KernelHardwareInfo hw_info;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(M), N, K, 1},
      {reinterpret_cast<const ElementA*>(a_ptr), stride_A,
       reinterpret_cast<const ElementB*>(b_ptr), stride_B},
      {{'{'}}{},
       reinterpret_cast<const ElementOut*>(residual_ptr), stride_C,
       reinterpret_cast<ElementOut*>(out_ptr), stride_D{{'}'}},
      hw_info};
  auto& fa = arguments.epilogue.thread;
  fa.alpha_ptr = s_alpha;   // per-row [M] (default dAlpha => per-row broadcast)
  fa.beta = {{ '1.0f' if has_residual else '0.0f' }};
  fa.bias_ptr = nullptr;    // linears are bias-free (=> bias contribution 0)

  Gemm gemm_op;
  static uint8_t* s_ws = nullptr;
  static size_t s_ws_sz = 0;
  size_t need = Gemm::get_workspace_size(arguments);
  if (need > s_ws_sz) {
    if (s_ws) cudaFree(s_ws);
    cudaMalloc(&s_ws, need);
    s_ws_sz = need;
  }
  AIT_CUTLASS_CHECK(gemm_op.can_implement(arguments));
  AIT_CUTLASS_CHECK(gemm_op.initialize(arguments, s_ws, stream));
  AIT_CUTLASS_CHECK(gemm_op.run(stream));
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, const void*, const void*, const void*, "
    "const void*, void*, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}}, {{b_ptr}}, {{scale_x_ptr}}, {{scale_w_ptr}},
{{indent}}    {{residual_ptr}}, {{out_ptr}}, {{m_expr}}, stream
{{indent}});
"""
)


def _tile_for_n(N):
    # swept on H200 for the go9 fp8 gemm shapes (RCR, cooperative FP8 FastAccum).
    if N >= 256 and N % 256 == 0:
        return "Shape<_128, _256, Shape<_128>>"
    if N > 128:
        return "Shape<_256, _128, Shape<_128>>"
    return "Shape<_128, _128, Shape<_128>>"


@registry.reg("cuda.gemm_rcr_fp8_fused.gen_function")
def gen_function(func_attrs):
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"], N=func_attrs["N"], K=func_attrs["K"],
        has_residual=func_attrs.get("has_residual", False),
        tile=_tile_for_n(func_attrs["N"]),
    )


@registry.reg("cuda.gemm_rcr_fp8_fused.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.gemm_rcr_fp8_fused.func_call")
def gen_function_call(func_attrs, indent="  "):
    ins = func_attrs["inputs"]
    a, b, scale_x, scale_w = ins[0], ins[1], ins[2], ins[3]
    residual_ptr = ins[4]._attrs["name"] if func_attrs.get("has_residual", False) else "nullptr"
    out = func_attrs["outputs"][0]
    m_expr = " * ".join(d._attrs["name"] for d in a._attrs["shape"][:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent, func_name=func_attrs["name"],
        a_ptr=a._attrs["name"], b_ptr=b._attrs["name"],
        scale_x_ptr=scale_x._attrs["name"], scale_w_ptr=scale_w._attrs["name"],
        residual_ptr=residual_ptr,
        out_ptr=out._attrs["name"], m_expr=m_expr,
    )
