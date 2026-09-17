#  Backend codegen for the SM100 MXFP8 (block-scaled e4m3) gemm: RowMajor-A / ColumnMajor-B,
#  A/B = mx_float8_t<e4m3> with per-32-element ue8m0 block scales (SFA activation, SFB weight)
#  applied INSIDE the tcgen05 UMMA, so the epilogue sees a real-scale f32 accumulator and only
#  does the (bias-free) residual add -> f16 out. No per-row alpha / dequant kernel (unlike the
#  per-tensor gemm_rcr_fp8_fused). N,K baked; M runtime. SM100 only.
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
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace {{func_name}}_ns {
using namespace cute;
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;  // e4m3 data + ue8m0 block scale
using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementOut = cutlass::half_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using TileShapeMNK    = {{tile}};
using ClusterShapeMNK = {{cluster}};
constexpr int AlignA = 16;  // 8-bit e4m3 -> 128-bit TMA
constexpr int AlignB = 16;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, TileShapeMNK, ClusterShapeMNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc,
    ElementOut, LayoutC, AlignC,
    ElementOut, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;  // default LinearCombination

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
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
using DataA   = typename ElementA::DataType;   // e4m3
using DataB   = typename ElementB::DataType;
using ScaleT  = typename ElementA::ScaleFactorType;  // ue8m0
using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

#ifndef AIT_CUTLASS_CHECK
#define AIT_CUTLASS_CHECK(status)                                              \\
  {                                                                            \\
    cutlass::Status s = (status);                                             \\
    if (s != cutlass::Status::kSuccess)                                        \\
      throw std::runtime_error(std::string("CUTLASS mxfp8 gemm error: ") +     \\
                               cutlassGetStatusString(s));                     \\
  }
#endif
}  // namespace {{func_name}}_ns

// A [M,{{K}}] e4m3 + SFA(ue8m0 swizzled), B [{{N}},{{K}}] e4m3 + SFB(ue8m0 swizzled) -> D [M,{{N}}] f16.
// Block scales are applied in the MMA; the epilogue only adds beta*residual (bias-free trunk).
void {{func_name}}(const void* a_ptr, const void* sfa_ptr, const void* b_ptr, const void* sfb_ptr,
                   const void* residual_ptr, void* out_ptr, int64_t M, cudaStream_t stream) {
  using namespace {{func_name}}_ns;
  const int N = {{N}}, K = {{K}};
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(M), K, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {static_cast<int>(M), N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {static_cast<int>(M), N, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(static_cast<int>(M), N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(static_cast<int>(M), N, K, 1));

  cutlass::KernelHardwareInfo hw_info;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(M), N, K, 1},
      {reinterpret_cast<const DataA*>(a_ptr), stride_A,
       reinterpret_cast<const DataB*>(b_ptr), stride_B,
       reinterpret_cast<const ScaleT*>(sfa_ptr), layout_SFA,
       reinterpret_cast<const ScaleT*>(sfb_ptr), layout_SFB},
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
    "\nvoid {{func_name}}(const void*, const void*, const void*, const void*, "
    "const void*, void*, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}}, {{sfa_ptr}}, {{b_ptr}}, {{sfb_ptr}},
{{indent}}    {{residual_ptr}}, {{out_ptr}}, {{m_expr}}, stream
{{indent}});
"""
)


def _tile_cluster():
    import os

    # 1SM default (MMA tile 128x128x128, K-tile a multiple of SFVecSize=32); 2SM opt-in.
    if os.environ.get("AIT_FP8_GEMM_2SM", "0") == "1":
        return "Shape<_256, _128, _128>", "Shape<_2, _1, _1>"
    return "Shape<_128, _128, _128>", "Shape<_1, _1, _1>"


@registry.reg("cuda.gemm_rcr_mxfp8.gen_function")
def gen_function(func_attrs):
    tile, cluster = _tile_cluster()
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"], N=func_attrs["N"], K=func_attrs["K"],
        has_residual=func_attrs.get("has_residual", False), tile=tile, cluster=cluster,
    )


@registry.reg("cuda.gemm_rcr_mxfp8.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.gemm_rcr_mxfp8.func_call")
def gen_function_call(func_attrs, indent="  "):
    ins = func_attrs["inputs"]
    a, sfa, b, sfb = ins[0], ins[1], ins[2], ins[3]
    residual_ptr = ins[4]._attrs["name"] if func_attrs.get("has_residual", False) else "nullptr"
    out = func_attrs["outputs"][0]
    m_expr = " * ".join(d._attrs["name"] for d in a._attrs["shape"][:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent, func_name=func_attrs["name"],
        a_ptr=a._attrs["name"], sfa_ptr=sfa._attrs["name"],
        b_ptr=b._attrs["name"], sfb_ptr=sfb._attrs["name"],
        residual_ptr=residual_ptr, out_ptr=out._attrs["name"], m_expr=m_expr,
    )
