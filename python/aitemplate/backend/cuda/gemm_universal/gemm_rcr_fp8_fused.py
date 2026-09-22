# CUTLASS SM90/SM100 FP8 GEMM with FP32 accumulation and fused dequantization:
# D_f16 = (scale_x[m] * scale_w[0]) * (A @ B.T) + beta * residual.
# N/K are static; M is dynamic. Scales are consumed directly by the epilogue.
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
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
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
// Architecture- and shape-specific tiles are selected by the code generator.
using TileShapeMNK    = {{tile}};
using ClusterShapeMNK = {{cluster}};
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;   // 16
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;   // 16
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value; // 8

// Per-token and per-weight scales are multiplied in FP32 before scaling the accumulator.
// These EVT visitors are supported by both SM90 and SM100 TMA epilogues.
using namespace cutlass::epilogue::fusion;
using ScaleProduct = Sm90EVT<Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    Sm90ColBroadcast<0, TileShapeMNK, float, float, Stride<_1,_0,int64_t>, 4>,
    Sm90ScalarBroadcast<float, Stride<_0,_0,int64_t>>>;
using ScaledAcc = Sm90EVT<Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    ScaleProduct, Sm90AccFetch>;
using FusionOp = Sm90EVT<Sm90Compute<cutlass::homogeneous_multiply_add, ElementOut, float, cutlass::FloatRoundStyle::round_to_nearest>,
    Sm90ScalarBroadcast<float, Stride<_0,_0,int64_t>>, Sm90SrcFetch<ElementOut>, ScaledAcc>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::{{arch_tag}}, cutlass::arch::OpClassTensorOp, TileShapeMNK, ClusterShapeMNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute,
    ElementOut, LayoutC, AlignC,
    ElementOut, LayoutC, AlignC,
    {{epi_sched}}, FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::{{arch_tag}}, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc, TileShapeMNK, ClusterShapeMNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    {{mainloop_sched}}>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue{{sched_arg}}>;
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

}  // namespace {{func_name}}_ns

// A [M,{{K}}] e4m3, B [{{N}},{{K}}] e4m3 -> D [M,{{N}}] f16 (fused per-row scale + residual)
void {{func_name}}(const void* a_ptr, const void* b_ptr, const void* scale_x_ptr,
                   const void* scale_w_ptr, const void* residual_ptr,
                   void* out_ptr, int64_t M, cudaStream_t stream) {
  using namespace {{func_name}}_ns;
  const int N = {{N}}, K = {{K}};
  // thread_local: one workspace set per host thread so concurrent inference threads (each on its
  // own stream) don't clobber a shared buffer (review finding #2). Costs the workspace x #threads.
  // Workspaces must retain their addresses across cached graph shapes.
  thread_local static bool workspace_initialized = false;
  const bool allocate_workspace = !workspace_initialized;

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
  // Compute sx[m] * sw[0] in FP32 inside the epilogue, removing the
  // scale-materialization kernel and its global-memory intermediate.
  fa = {% raw %}{
    {{ {% endraw %}{{ '1.0f' if has_residual else '0.0f' }}{% raw %} }, {nullptr}, {}}, {},
    {{{reinterpret_cast<const float*>(scale_x_ptr), 0.f, {}},
      {{0.f}, {reinterpret_cast<const float*>(scale_w_ptr)}, {}}, {}}, {}, {}}, {}};{% endraw %}

  Gemm gemm_op;
  thread_local static uint8_t* s_ws = nullptr;
  if (allocate_workspace) {
    auto size_args = arguments;
    size_args.problem_shape = { {{max_m}}, N, K, 1 };
    size_t need = Gemm::get_workspace_size(size_args);
    if (need > 0) cudaMalloc(&s_ws, need);
    workspace_initialized = true;
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
    # SM90: swept on H200 for the trunk fp8 gemm shapes (RCR, cooperative FP8 FastAccum).
    # Avoid padding these channel widths to a multiple of 128.
    if N % 192 == 0:
        return "Shape<_128, _192, Shape<_128>>"
    if N >= 256 and N % 256 == 0:
        return "Shape<_128, _256, Shape<_128>>"
    if N > 128:
        return "Shape<_256, _128, Shape<_128>>"
    return "Shape<_128, _128, Shape<_128>>"


# Base collective configurations. gen_function adds the measured shape/M dispatch.
# SM90 uses cooperative FP8 FastAccum; SM100 selects 1SM/2SM via the cluster
# shape and uses the CLC scheduler (the extra void kernel argument).
def _arch_config(func_attrs):
    from aitemplate.backend.target import Target

    import os

    arch = Target.current()._arch
    if arch == "100":
        # Auto+Auto is the guaranteed-compatible SM100 pairing (cutlass example 70 ships it
        # with a per-row fusion). ClusterShape M parity selects the schedule under Auto:
        # <_1,_1,_1> -> 1SM (MMA tile M=128), <_2,_1,_1> -> 2SM (MMA tile M=256, 2x MMA
        # throughput). Explicit AIT_FP8_GEMM_2SM overrides disable automatic dispatch;
        # gen_function otherwise combines 1SM and 2SM for supported projection shapes.
        if os.environ.get("AIT_FP8_GEMM_2SM", "0") == "1":
            tile, cluster = "Shape<_256, _128, _64>", "Shape<_2, _1, _1>"
        else:
            tile, cluster = "Shape<_128, _128, _64>", "Shape<_1, _1, _1>"
        return {
            "arch_tag": "Sm100",
            "epi_sched": "cutlass::epilogue::collective::EpilogueScheduleAuto",
            "mainloop_sched": "cutlass::gemm::collective::KernelScheduleAuto",
            "tile": tile,
            "cluster": cluster,
            "sched_arg": ", void",
        }
    return {
        "arch_tag": "Sm90",
        "epi_sched": "cutlass::epilogue::TmaWarpSpecializedCooperative",
        "mainloop_sched": "cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum",
        "tile": _tile_for_n(func_attrs["N"]),
        "cluster": "Shape<_1, _1, _1>",
        "sched_arg": "",
    }


@registry.reg("cuda.gemm_rcr_fp8_fused.gen_function")
def gen_function(func_attrs):
    import os
    from aitemplate.backend.target import Target

    name = func_attrs["name"]
    config = _arch_config(func_attrs)
    max_m = 1
    for dim in func_attrs["inputs"][0]._attrs["shape"][:-1]:
        max_m *= dim._attrs["values"][-1]
    kwargs = dict(N=func_attrs["N"], K=func_attrs["K"], max_m=max_m,
                  has_residual=func_attrs.get("has_residual", False))
    # H200 graph microbenchmarks: narrow N tiles fill the SMs at small M;
    # larger M tiles help the wide projections once multiple waves are needed.
    # Keep these measured choices scoped to the tested projection dimensions.
    projection_shapes = {(192, 192), (192, 384), (192, 576),
                         (384, 192), (576, 192), (1152, 192)}
    if Target.current()._arch == "90" and (kwargs["N"], kwargs["K"]) in projection_shapes:
        n = kwargs["N"]
        if n == 192:
            choices = [(5632, 128, 64), (8192, 128, 128), (None, 128, 192)]
        elif n == 384:
            choices = [(2816, 128, 64), (5632, 128, 128), (None, 128, 192)]
        elif n == 576:
            choices = [(1792, 128, 64), (3328, 128, 128),
                       (8192, 128, 192), (None, 256, 192)]
        else:
            choices = [(896, 128, 64), (10240, 128, 192), (None, 256, 192)]
        kernels, calls = [], []
        for index, (limit, tile_m, tile_n) in enumerate(choices):
            variant = name + "_m" + str(index)
            tuned = dict(config, tile=f"Shape<_{tile_m}, _{tile_n}, _128>")
            kernels.append(FUNC_TEMPLATE.render(func_name=variant, **kwargs, **tuned))
            call = f"{variant}(a, b, sx, sw, residual, out, M, stream);"
            calls.append(f"  if (M <= {limit}) {{ {call} return; }}" if limit else "  " + call)
        dispatch = (f"\nvoid {name}(const void* a, const void* b, const void* sx, const void* sw, "
                    "const void* residual, void* out, int64_t M, cudaStream_t stream) {\n"
                    + "\n".join(calls) + "\n}\n")
        return "\n".join(kernels) + dispatch
    # Small M favors the original 1SM tile; channel-aligned 2SM tiles win at
    # large M. Keep the explicit 2SM environment override authoritative.
    if (Target.current()._arch == "100" and func_attrs["N"] % 192 == 0
            and "AIT_FP8_GEMM_2SM" not in os.environ):
        tuned_shape = (kwargs["N"], kwargs["K"]) in projection_shapes
        tiny_limit = ({192: 6144, 384: 3072, 576: 2048, 1152: 1024}[kwargs["N"]]
                      if tuned_shape else 0)
        tiny = ""
        tiny_call = ""
        if tiny_limit:
            tiny = FUNC_TEMPLATE.render(func_name=name + "_tiny", **kwargs,
                                        **dict(config, tile="Shape<_128, _64, _64>"))
            tiny_call = f"  if (M <= {tiny_limit}) {{ {name}_tiny(a, b, sx, sw, residual, out, M, stream); return; }}\n"
        small_config = dict(config)
        if tuned_shape and kwargs["N"] in (192, 576):
            small_config["tile"] = "Shape<_128, _96, _64>"
        small = FUNC_TEMPLATE.render(func_name=name + "_small", **kwargs, **small_config)
        large_k = 128 if (kwargs["N"], kwargs["K"]) == (192, 384) else 64
        large_config = dict(config, tile=f"Shape<_256, _192, _{large_k}>",
                            cluster="Shape<_2, _1, _1>")
        large = FUNC_TEMPLATE.render(func_name=name + "_large", **kwargs, **large_config)
        dispatch = f"""
void {name}(const void* a, const void* b, const void* sx, const void* sw,
            const void* residual, void* out, int64_t M, cudaStream_t stream) {{
{tiny_call}  if (M < 8192) {{
    {name}_small(a, b, sx, sw, residual, out, M, stream);
  }} else {{
    {name}_large(a, b, sx, sw, residual, out, M, stream);
  }}
}}
"""
        return tiny + small + large + dispatch
    return FUNC_TEMPLATE.render(func_name=name, **kwargs, **config)


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
