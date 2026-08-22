#  Backend codegen for the FUSED fp8 conv (conv2d_fp8): CUTLASS 3x SM90 fp8 implicit-gemm
#  conv (fprop) whose epilogue folds dequant+bias+residual+relu:
#    D_f16 = act( alpha * conv(xq,w) + beta*residual + bias[k] ),  alpha = scale_x*scale_w
#  When emit_amax: a custom FusionCallbacks wraps LinCombPerColBiasEltAct with an
#  Sm90ScalarReduction<amax> root, so the epilogue ALSO computes amax(output) into a scalar
#  (for the next conv's single-pass quantize). Writes f16 directly (no f32 acc round-trip).
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"

// Custom fusion: LinCombPerColBiasEltAct + amax(D). Defined once per TU (guarded).
{% raw %}
#ifndef AIT_FP8_LINCOMB_PERCOL_AMAX
#define AIT_FP8_LINCOMB_PERCOL_AMAX
namespace cutlass::epilogue::fusion {
template <template <class> class ActivationFn_, class ElementOutput_, class ElementCompute_,
          class ElementAmax_, class ElementBias_, class ElementSource_, class ElementScalar_,
          int AlignmentBias_, FloatRoundStyle RoundStyle_>
struct LinCombPerColBiasEltActAmaxD
    : LinCombPerColBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_,
                              ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementAmax = ElementAmax_;
};
template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          template <class> class ActivationFn, class ElementOutput, class ElementCompute,
          class ElementAmax, class ElementBias, class ElementSource, class ElementScalar,
          int AlignmentBias, FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    LinCombPerColBiasEltActAmaxD<ActivationFn, ElementOutput, ElementCompute, ElementAmax,
                                 ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile>
    : Sm90EVT<Sm90ScalarReduction<detail::amax, cutlass::atomic_maximum, ElementAmax, ElementCompute, RoundStyle>,
              Sm90LinCombPerColBiasEltAct<StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn,
                                          ElementOutput, ElementCompute, ElementBias, ElementSource,
                                          ElementScalar, AlignmentBias, RoundStyle>> {
  using Impl = Sm90EVT<Sm90ScalarReduction<detail::amax, cutlass::atomic_maximum, ElementAmax, ElementCompute, RoundStyle>,
              Sm90LinCombPerColBiasEltAct<StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn,
                                          ElementOutput, ElementCompute, ElementBias, ElementSource,
                                          ElementScalar, AlignmentBias, RoundStyle>>;
  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    using StrideAlpha = Stride<_0, _0, int64_t>;
    using StrideBeta = Stride<_0, _0, int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta dBeta = {_0{}, _0{}, 0};
    using StrideBias = Stride<_0, _1, int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    using ActivationArguments =
        typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();
    ElementAmax* amax_ptr = nullptr;
    operator typename Impl::Arguments() const {
      return {
          {  // child: Sm90LinCombPerColBiasEltAct = { ternary, activation }
              {  // ternary: beta*C + (alpha*acc + bias)
                  {{beta}, {beta_ptr}, {dBeta}},
                  {},
                  {  // ternary: alpha*acc + bias
                      {{alpha}, {alpha_ptr}, {dAlpha}},
                      {},
                      {bias_ptr, ElementBias(0), dBias},
                      {}},
                  {}},
              activation},
          {amax_ptr}};
    }
  };
  using Impl::Impl;
};
}  // namespace cutlass::epilogue::fusion
#endif  // AIT_FP8_LINCOMB_PERCOL_AMAX
{% endraw %}

namespace {{func_name}}_ns {
using namespace cute;

using ElementAct     = cutlass::float_e4m3_t;
using ElementFlt     = cutlass::float_e4m3_t;
using ElementOut     = cutlass::half_t;
using ElementAcc     = float;
using ElementCompute = float;
using TileShapeMNK    = Shape<_128, _128, Shape<_128>>;
using ClusterShapeMNK = Shape<_1, _1, _1>;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementAct>::value;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementFlt>::value;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value;

using FusionOp =
{% if emit_amax %}
    cutlass::epilogue::fusion::LinCombPerColBiasEltActAmaxD<
        {{act_fn}}, ElementOut, ElementCompute, float, ElementOut, ElementOut, float,
        AlignC, cutlass::FloatRoundStyle::round_to_nearest>;
{% else %}
    cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
        {{act_fn}}, ElementOut, ElementCompute, ElementOut, ElementOut, float>;
{% endif %}

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShapeMNK, ClusterShapeMNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute,
    ElementOut, cutlass::layout::TensorNHWC, AlignC,
    ElementOut, cutlass::layout::TensorNHWC, AlignC,
    cutlass::epilogue::TmaWarpSpecialized, FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::conv::Operator::kFprop,
    ElementAct, cutlass::layout::TensorNHWC, AlignA,
    ElementFlt, cutlass::layout::TensorNHWC, AlignB,
    ElementAcc, TileShapeMNK, ClusterShapeMNK,
    cutlass::conv::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::conv::collective::KernelScheduleAuto>::CollectiveOp;

using ProblemShape = cutlass::conv::ConvProblemShape<
    CollectiveMainloop::DispatchPolicy::ConvOp,
    CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
using ConvKernel = cutlass::conv::kernel::ConvUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

#ifndef AIT_CUTLASS_CHECK
#define AIT_CUTLASS_CHECK(status)                                              \\
  {                                                                            \\
    cutlass::Status s = (status);                                             \\
    if (s != cutlass::Status::kSuccess)                                        \\
      throw std::runtime_error(std::string("CUTLASS conv error: ") +          \\
                               cutlassGetStatusString(s));                     \\
  }
#endif

// alpha[0] = scale_x*scale_w; also zero the output-amax buffer (before the conv's atomicMax).
__global__ void alpha_kernel(const float* a, const float* b, float* alpha_o, float* amax_o) {
  if (threadIdx.x == 0) {
    alpha_o[0] = a[0] * b[0];
    if (amax_o) amax_o[0] = 0.f;
  }
}
}  // namespace {{func_name}}_ns

void {{func_name}}(const void* in_ptr, const void* weight_ptr, const void* scale_x_ptr,
                   const void* scale_w_ptr, const void* bias_ptr, const void* residual_ptr,
                   void* out_ptr, void* amax_out_ptr, int64_t N, cudaStream_t stream) {
  using namespace {{func_name}}_ns;
  static float* s_alpha = nullptr;
  if (!s_alpha) cudaMalloc(&s_alpha, sizeof(float));
  alpha_kernel<<<1, 32, 0, stream>>>(
      reinterpret_cast<const float*>(scale_x_ptr),
      reinterpret_cast<const float*>(scale_w_ptr), s_alpha,
      reinterpret_cast<float*>(amax_out_ptr));

  ProblemShape problem_shape(
      cutlass::conv::Mode::kCrossCorrelation,
      {static_cast<int>(N), {{H}}, {{W}}, {{C}}},
      {{'{'}}{{K}}, {{R}}, {{S}}, {{C}}{{'}'}},
      {{'{'}}{{pad}}, {{pad}}{{'}'}}, {{'{'}}{{pad}}, {{pad}}{{'}'}},
      {{'{'}}{{stride}}, {{stride}}{{'}'}}, {1, 1}, 1);

  typename Conv::ConvKernel::StrideC stride_C;
  typename Conv::ConvKernel::StrideD stride_D;
  for_each(make_seq<rank<0>(typename Conv::ConvKernel::StrideC{})>{}, [&](auto i) {
    get<0, i>(stride_C) = problem_shape.stride_C[ProblemShape::RankT - 2 - i];
  });
  for_each(make_seq<rank<0>(typename Conv::ConvKernel::StrideD{})>{}, [&](auto i) {
    get<0, i>(stride_D) = problem_shape.stride_C[ProblemShape::RankT - 2 - i];
  });

  typename Conv::Arguments arguments{
      problem_shape,
      {reinterpret_cast<const ElementAct*>(in_ptr),
       reinterpret_cast<const ElementFlt*>(weight_ptr)},
      {{'{'}}{},
       reinterpret_cast<const ElementOut*>(residual_ptr), stride_C,
       reinterpret_cast<ElementOut*>(out_ptr), stride_D{{'}'}}};
  auto& fa = arguments.epilogue.thread;
  fa.alpha_ptr = s_alpha;
  fa.beta = {{ '1.0f' if has_residual else '0.0f' }};
  fa.bias_ptr = reinterpret_cast<const ElementOut*>(bias_ptr);
{% if emit_amax %}
  fa.amax_ptr = reinterpret_cast<float*>(amax_out_ptr);
{% endif %}

  Conv conv_op;
  static uint8_t* s_ws = nullptr;
  static size_t s_ws_sz = 0;
  size_t need = Conv::get_workspace_size(arguments);
  if (need > s_ws_sz) {
    if (s_ws) cudaFree(s_ws);
    cudaMalloc(&s_ws, need);
    s_ws_sz = need;
  }
  AIT_CUTLASS_CHECK(conv_op.can_implement(arguments));
  AIT_CUTLASS_CHECK(conv_op.initialize(arguments, s_ws, stream));
  AIT_CUTLASS_CHECK(conv_op.run(stream));
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, const void*, const void*, const void*, const void*, "
    "const void*, void*, void*, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}}, {{weight_ptr}}, {{scale_x_ptr}}, {{scale_w_ptr}}, {{bias_ptr}},
{{indent}}    {{residual_ptr}}, {{out_ptr}}, {{amax_out_ptr}}, {{n_expr}}, stream
{{indent}});
"""
)


@registry.reg("cuda.conv2d_fp8.gen_function")
def gen_function(func_attrs):
    act_fn = (
        "cutlass::epilogue::thread::ReLU"
        if func_attrs["relu"]
        else "cutlass::epilogue::thread::Identity"
    )
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"], act_fn=act_fn,
        has_residual=func_attrs.get("has_residual", False),
        emit_amax=func_attrs.get("emit_amax", False),
        H=func_attrs["H"], W=func_attrs["W"], C=func_attrs["C"],
        K=func_attrs["K"], R=func_attrs["R"], S=func_attrs["S"],
        OH=func_attrs["OH"], OW=func_attrs["OW"],
        pad=func_attrs["pad"], stride=func_attrs["stride"],
    )


@registry.reg("cuda.conv2d_fp8.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.conv2d_fp8.func_call")
def gen_function_call(func_attrs, indent="  "):
    ins = func_attrs["inputs"]
    x, w, scale_x, scale_w, bias = ins[0], ins[1], ins[2], ins[3], ins[4]
    residual_ptr = ins[5]._attrs["name"] if func_attrs.get("has_residual", False) else "nullptr"
    outs = func_attrs["outputs"]
    amax_out_ptr = outs[1]._attrs["name"] if func_attrs.get("emit_amax", False) else "nullptr"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent, func_name=func_attrs["name"],
        in_ptr=x._attrs["name"], weight_ptr=w._attrs["name"],
        scale_x_ptr=scale_x._attrs["name"], scale_w_ptr=scale_w._attrs["name"],
        bias_ptr=bias._attrs["name"], residual_ptr=residual_ptr,
        out_ptr=outs[0]._attrs["name"], amax_out_ptr=amax_out_ptr,
        n_expr=x._attrs["shape"][0]._attrs["name"],
    )
