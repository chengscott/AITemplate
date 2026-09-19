#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
GEMM Specialization for
C = UnaryOp2(BinaryOp2(BinaryOp1(UnaryOp1(GeMM(A, B) + bias), D1), D2)),
"""

import re
from functools import partial

import jinja2
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import gemm_common
from aitemplate.backend.cuda.gemm_universal import common, gemm_rcr
from aitemplate.backend.target import Target

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


# For config extraction.
GEMM_UNIVERSAL_WITH_BROADCAST_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        {{elem_type}}, {{layout.cutlass_layout_a}},
        {{elem_type}}, {{layout.cutlass_layout_b}},
        {{elem_type}}, {{layout.cutlass_layout_c}},
        {{acc_type}},
        cutlass::arch::OpClassTensorOp,
        {{arch}},
        {{tb_shape}},
        {{warp_shape}},
        {{instruction_shape}},
        {{epilogue_functor}}<
            {{elem_type}}, {{acc_type}}, {{acc_type}},
            {{elem_type}}, {{epilogue_vector_length}},
            {{unary_op1}}, {{binary_op1}}, {{unary_op2}}
{% if has_d1 %}
            , {{binary_op2}}
{% endif %}
        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        {{stage}},
        {{alignment_a}},
        {{alignment_b}}
    >;
"""
)

# we use the transposed problem with swapped A and B
# operands and column-major C and D in CUTLASS 3.x
EPILOGUE_TENSOR_BROADCAST_TEMPLATE = jinja2.Template(
    """
using {{epilogue_name}} =
  cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueTensorBroadcast<
      cutlass::gemm::TagToStrideC_t<cutlass::layout::LayoutTranspose<{{layout_c}}>::type>,
      cutlass::gemm::TagToStrideC_t<cutlass::layout::LayoutTranspose<{{layout_d}}>::type>,
      cutlass::epilogue::thread::LinearCombinationTensorBroadcast<
        {{element_d}}, {{element_accumulator}}, {{element_compute}}, {{element_c}},
        {{unary_op1}},
        {{binary_op1}},
{% if has_d1 %}
        {{binary_op2}},
{% else %}
        cutlass::epilogue::thread::detail::NoOp,
{% endif %}
        {{unary_op2}}
        >,
      {{epilogue_schedule}}>>;
"""
)

# For func codegen.
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    {
        static_cast<coord_t>({{layout.m}}),
        static_cast<coord_t>({{layout.n}}),
        static_cast<coord_t>({{layout.k}})
    },                                                       // GemmCoord problem_size
{% if support_split_k %}
    split_k,                                                 // int batch_count
{% else %}
    1,                                                       // int batch_count
{% endif %}
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,          // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,          // void const * ptr_B
    ({{elem_output_type}}*)(d0_ptr),                         // void const * ptr_C1
{% if has_d1 %}
    ({{elem_output_type}}*)(d1_ptr),                         // void const * ptr_C2
{% endif %}
    ({{elem_output_type}}*) (c_ptr) + output_offset,         // void * ptr_D
    ({{elem_input_type}}*) (bias_ptr),                       // void * ptr_Vector
    nullptr,                                                 // void * ptr_Tensor
    input_a_batch_stride,                                    // int64_t batch_stride_A
    input_b_batch_stride,                                    // int64_t batch_stride_B
    0,                                                       // int64_t batch_stride_C1
{% if has_d1 %}
    0,                                                       // int64_t batch_stride_C2
{% endif %}
    0,                                                       // int64_t batch_stride_D
    0,                                                       // int64_t batch_stride_Vector
    0,                                                       // int64_t batch_stride_Tensor
    input_a_stride,                                          // typename LayoutA::Stride::Index lda
    input_b_stride,                                          // typename LayoutB::Stride::Index ldb
    {{layout.stride_c}},                                     // typename LayoutC::Stride::Index ldc1
{% if has_d1 %}
    {{layout.stride_c}},                                     // typename LayoutC::Stride::Index ldc2
{% endif %}
    output_stride,                                           // typename LayoutC::Stride::Index ldd
    0,                                                       // typename LayoutC::Stride::Index ldr
    0,                                                       // typename LayoutC::Stride::Index ldt
"""
)

# we use the transposed problem with swapped A and B
# operands and column-major C and D in CUTLASS 3.x
PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>({{layout.n}}),
        static_cast<coord_t>({{layout.m}}),
        static_cast<coord_t>({{layout.k}}),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    {  // MainloopArguments mainloop (explicit brace: mma_promotion_interval defaults)
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,              // ElementA const* ptr_A
    {input_b_stride, cute::Int<1>{}, cute::Int<0>{}},            // StrideA dA
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,              // ElementB const* ptr_B
    {input_a_stride, cute::Int<1>{}, cute::Int<0>{}},            // StrideB dB
    },
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename ThreadEpilogueOp::Params thread
        {cute::Int<1>{}, {{layout.stride_c}}, cute::Int<0>{}},   // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD* ptr_D
        {cute::Int<1>{}, output_stride, cute::Int<0>{}},         // StrideD dD
        ({{elem_input_type}}*)(bias_ptr),                        // ElementBias* ptr_Bias
        ({{elem_output_type}}*)(d0_ptr),                         // ElementC* ptr_C0
{% if has_d1 %}
        ({{elem_output_type}}*)(d1_ptr),                         // ElementC* ptr_C1
{% else %}
        nullptr,
{% endif %}
    },                                                           // EpilogueArguments epilogue
"""
)

# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    {
        static_cast<coord_t>({{layout.m}}),
        static_cast<coord_t>({{layout.n}}),
        static_cast<coord_t>({{layout.k}})
    },                                                       // GemmCoord problem_size
{% if support_split_k %}
    split_k,                                                 // int batch_count
{% else %}
    1,                                                       // int batch_count
{% endif %}
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*) a_ptr,                            // void const * ptr_A
    ({{elem_input_type}}*) b_ptr,                            // void const * ptr_B
    ({{elem_output_type}}*) d0_ptr,                          // void const * ptr_C1
{% if has_d1 %}
    ({{elem_output_type}}*) d1_ptr,                          // void const * ptr_C2
{% endif %}
    ({{elem_output_type}}*) (c_ptr) + output_offset,         // void * ptr_D
    ({{elem_input_type}}*) bias_ptr,                         // void * ptr_Vector
    nullptr,                                                 // void * ptr_Tensor
    0,                                                       // int64_t batch_stride_A
    0,                                                       // int64_t batch_stride_B
    0,                                                       // int64_t batch_stride_C1
{% if has_d1 %}
    0,                                                       // int64_t batch_stride_C2
{% endif %}
    0,                                                       // int64_t batch_stride_D
    0,                                                       // int64_t batch_stride_Vector
    0,                                                       // int64_t batch_stride_Tensor
    {{layout.stride_a}},                                     // typename LayoutA::Stride::Index lda
    {{layout.stride_b}},                                     // typename LayoutB::Stride::Index ldb
    {{layout.stride_c}},                                     // typename LayoutC::Stride::Index ldc1
{% if has_d1 %}
    {{layout.stride_c}},                                     // typename LayoutC::Stride::Index ldc2
{% endif %}
    output_stride,                                           // typename LayoutC::Stride::Index ldd
    0,                                                       // typename LayoutC::Stride::Index ldr
    0,                                                       // typename LayoutC::Stride::Index ldt
"""
)

# we use the transposed problem with swapped A and B
# operands and column-major C and D in CUTLASS 3.x
PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>({{layout.n}}),
        static_cast<coord_t>({{layout.m}}),
        static_cast<coord_t>({{layout.k}}),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    {  // MainloopArguments mainloop (explicit brace: mma_promotion_interval defaults)
    ({{elem_input_type}}*)(b_ptr),                               // ElementB const* ptr_A
    { {{layout.stride_b}}, cute::Int<1>{}, cute::Int<0>{}},      // StrideB dA
    ({{elem_input_type}}*)(a_ptr),                               // ElementA const* ptr_B
    { {{layout.stride_a}}, cute::Int<1>{}, cute::Int<0>{}},      // StrideA dB
    },
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename ThreadEpilogueOp::Params thread
        {cute::Int<1>{}, {{layout.stride_c}}, cute::Int<0>{}},   // StrideC dC
        ({{elem_output_type}}*)(c_ptr),                          // ElementD* ptr_D
        {cute::Int<1>{}, output_stride, cute::Int<0>{}},         // StrideD dD
        ({{elem_input_type}}*)(bias_ptr),                        // ElementBias* ptr_Bias
        ({{elem_output_type}}*)(d0_ptr),                         // ElementC* ptr_C0
{% if has_d1 %}
        ({{elem_output_type}}*)(d1_ptr),                         // ElementC* ptr_C1
{% else %}
        nullptr,
{% endif %}
    },                                                           // EpilogueArguments epilogue
"""
)

# SM100 (Blackwell) residual/gate EVT. The single-source broadcast configs (binary_op2 is None,
# so exactly one auxiliary tensor d0, carried through the epilogue C operand) map onto EVT:
#   * add / add_relu -> the predefined LinCombPerColBias / LinCombPerColBiasEltAct<Act> functor
#     (D = act(alpha*acc + beta*d0 + per-col bias), beta = 1).
#   * mul -> D = (alpha*acc + per-col bias) * d0. No predefined functor, but the EVT *tree* is
#     the LinCombPerColBias tree with the beta*C add swapped for a top multiplies(inner, C) node.
#     We supply it via a custom FusionOperation tag + a Sm90 FusionCallbacks specialization
#     (SM100 callbacks alias to the Sm90 ones, so one specialization covers both); the builder
#     injects CtaTileShapeMNK, so the Sm90RowBroadcast bias node needs no hand-derived tile.
# add_add / mul_add (binary_op2 set -> a *second* aux tensor d1) need an Sm90AuxLoad node, whose
# TMA smem-layout-atom/stages template args the CollectiveBuilder derives internally and are very
# fragile to hand-emit; those stay on the SM80 fallback.
_UNSUPPORTED_SM100 = object()

_IDENTITY = "cutlass::epilogue::thread::Identity"
_RELU = "cutlass::epilogue::thread::ReLu"


def _residual_evt_arches():
    """Arches on which gemm_rcr_bias_add / _relu route through the LinCombPerColBias EVT.

    SM100 (Blackwell) always: without it these residual gemms fall back to the SM80 2.x
    kernels, and the tcgen05 EVT is up to ~2x faster on memory-bound, small-N gemms.

    SM90 (Hopper) is OPT-IN via AIT_SM90_RESIDUAL_EVT=1 (default off). It builds + parity-
    checks correctly and its SM90 3.x EVT candidates are profiled, but on memory-bound
    small-N gemms the SM90 TMA warp-specialized kernels are not faster than SM80 (unlike
    Blackwell tcgen05), so the profiler keeps SM80 -- off by default (no build cost),
    available for a compute-bound regime where the SM90 3.x kernels could win.
    """
    import os

    if os.environ.get("AIT_SM90_RESIDUAL_EVT") == "1":
        return ("90", "100")
    return ("100",)


def _sm100_evt_kind(unary_op1, binary_op1, binary_op2, unary_op2):
    """Classify a broadcast config into an SM100-EVT-supported kind, else _UNSUPPORTED_SM100.

    Returns (kind, activation_tag): kind in {"add", "mul"}; activation_tag is the outer
    activation tag (None, or e.g. ReLu) for the "add" flat-functor path.
    """
    if unary_op1 != _IDENTITY or binary_op2 is not None:
        return _UNSUPPORTED_SM100  # inner activation / second aux tensor -> not single-source EVT
    if binary_op1 == "cutlass::plus":
        if unary_op2 == _IDENTITY:
            return ("add", None)
        if unary_op2 == _RELU:
            return ("add", _RELU)
        return _UNSUPPORTED_SM100
    if binary_op1 == "cutlass::multiplies":
        if unary_op2 == _IDENTITY:
            return ("mul", None)
        return _UNSUPPORTED_SM100  # mul_tanh / sigmoid_mul need extra unary nodes
    return _UNSUPPORTED_SM100


class _LinCombPerColBiasMulFunctor:
    """SM100/SM90 EVT epilogue fusion: D = (alpha*acc + per-column bias) * C, with C = d0.

    The multiply counterpart of gemm_rcr_bias._LinCombPerColBiasFunctor. There is no predefined
    cutlass fusion Operation for it, so this emits a custom tag whose FusionCallbacks is defined
    by SM100_MUL_EVT_HEADER (injected into the .cu). Runtime args are identical to plain
    LinCombPerColBias (alpha, bias, ptr_C=d0), so the SM100 residual problem-args template is reused.
    """

    value = 10102  # profile-cache key sentinel

    def __init__(
        self, element_output, element_compute, element_bias, element_source, element_scalar
    ):
        self.element_output = element_output
        self.element_compute = element_compute
        self.element_bias = element_bias
        self.element_source = element_source
        self.element_scalar = element_scalar

    def emit_declaration(self):
        return (
            "cutlass::epilogue::fusion::LinCombPerColBiasMul<"
            f"{self.element_output}, {self.element_compute}, {self.element_bias}, "
            f"{self.element_source}, {self.element_scalar}>"
        )


# Custom fusion op for gemm_rcr_bias_mul on SM100/SM90: the LinCombPerColBias EVT tree with the
# top "beta*C + inner" multiply_add replaced by "inner * C" (C = d0, the gate/up tensor). Only a
# Sm90 FusionCallbacks specialization is needed -- sm100_callbacks_tma_warpspecialized.hpp aliases
# every Sm100TmaWarpSpecialized callback to its Sm90 counterpart. Include-guarded so it can be
# prepended to each emitted instance safely.
SM100_MUL_EVT_HEADER = """
#ifndef AIT_LINCOMB_PERCOL_BIAS_MUL_DEFINED
#define AIT_LINCOMB_PERCOL_BIAS_MUL_DEFINED
namespace cutlass { namespace epilogue { namespace fusion {

// FusionOperation tag (inherits LinCombPerColBias fields; builder treats it as tagged so it
// injects CtaTileShapeMNK into the callbacks below).
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasMul
    : LinCombPerColBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {};

// EVT tree: (alpha*acc + per-col bias) * C
template<
  int StagesC, class CtaTileShapeMNK, class EpilogueTile,
  class ElementOutput, class ElementCompute,
  class ElementBias = ElementOutput, class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColBiasMul =
  Sm90EVT<Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, RoundStyle>, // inner * C
    Sm90EVT<Sm90Compute<cutlass::homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha*acc + bias
      Sm90ScalarBroadcast<ElementScalar, cute::Stride<cute::_0,cute::_0,int64_t>>,
      Sm90AccFetch,
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, cute::Stride<cute::_0,cute::_1,int64_t>, AlignmentBias>
    >,
    Sm90SrcFetch<ElementSource> // C = d0
  >;

template <
  int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
  class ElementOutput, class ElementCompute, class ElementBias, class ElementSource, class ElementScalar,
  int AlignmentBias, FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    LinCombPerColBiasMul<ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerColBiasMul<
      StagesC, CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle> {
  using Impl = Sm90LinCombPerColBiasMul<
    StagesC, CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;
  using Operation = LinCombPerColBiasMul<
    ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = cute::Stride<cute::_0,cute::_0,int64_t>;
    using StrideBeta  = cute::Stride<cute::_0,cute::_0,int64_t>;
    StrideAlpha dAlpha = {cute::_0{}, cute::_0{}, 0};
    StrideBeta  dBeta  = {cute::_0{}, cute::_0{}, 0};

    using StrideBias = cute::Stride<cute::_0,cute::_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {                                          // multiplies : inner * C
          {                                        // multiply_add : alpha*acc + bias
            {{alpha}, {alpha_ptr}, {dAlpha}},      // leaf args : alpha
            {},                                    // leaf args : acc
            {bias_ptr, ElementBias(0), dBias},     // leaf args : bias
            {}                                     // ternary args : multiply_add
          },
          {},                                      // leaf args : C (d0)
          {}                                       // binary args : multiplies
        };
    }
  };

  using Impl::Impl;
};

}}}
#endif  // AIT_LINCOMB_PERCOL_BIAS_MUL_DEFINED
"""


# SM100 non-transposed problem args for the residual-add EVT (LinCombPerColBias[EltAct]).
# Same runtime args for add and add_relu (activation is compile-time in the functor); the
# residual d0 is the source C (beta = 1), the per-col bias broadcasts down the columns.
SM100_RESIDUAL_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    {  // MainloopArguments mainloop (non-transposed; bias+residual fused via EVT)
    ({{elem_input_type}}*)(a_ptr),                              // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                        // StrideA dA
    ({{elem_input_type}}*)(b_ptr),                              // ElementB const* ptr_B
    {K, cute::Int<1>{}, cute::Int<0>{}},                        // StrideB dB
    },
    {  // EpilogueArguments (act(alpha*acc + beta*d0 + bias))
        {                                                        // thread (fusion args)
            ElementComputeEpilogue(1),                           // alpha
            ElementComputeEpilogue(1),                           // beta (residual C = d0)
            nullptr,                                             // alpha_ptr
            nullptr,                                             // beta_ptr
            {cute::Int<0>{}, cute::Int<0>{}, int64_t(0)},        // StrideAlpha dAlpha
            {cute::Int<0>{}, cute::Int<0>{}, int64_t(0)},        // StrideBeta dBeta
            ({{elem_input_type}}*)(bias_ptr),                    // ElementBias const* bias_ptr
            {cute::Int<0>{}, cute::Int<1>{}, int64_t(0)},        // StrideBias dBias (per-col)
        },
        ({{elem_output_type}}*)(d0_ptr),                         // ElementC const* ptr_C (residual)
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideC dC (row-major MxN)
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/epilogue_tensor_broadcast.hpp"
#include "cutlass/epilogue/thread/linear_combination_tensor_broadcast.hpp"
// SM100 residual EVT (gemm_rcr_bias_add / _relu -> LinCombPerColBias[EltAct]) is built via
// the 3.x EPILOGUE collective builder + fusion operations. The broadcast profiler/function
// header set was written for the 2.x/SM90 TensorBroadcast path and omitted these, so the
// emitted `cutlass::epilogue::collective::CollectiveBuilder<...>` errored with
// "namespace has no member CollectiveBuilder". (The plain gemm_rcr_bias path already
// includes the epilogue collective builder.)
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

using bfloat16 = nv_bfloat16;

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \\
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \\
      std::cerr << msg << std::endl;                                                  \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}

{% if is_profiler %}
template <typename GemmInstance>
void {{function_name}} (
    GemmInstance& gemm_op,
{% else %}
void {{function_name}} (
{% endif %}
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* d0_ptr,
{% if has_d1 %}
    void* d1_ptr,
{% endif %}
    void* c_ptr,
    uint8_t* workspace,
{% if support_split_k %}
    int split_k,
{% endif %}
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
    cudaStream_t stream
  ) {
  {{shape_eval}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{extra_shape}}
  {{input_output_checks}}

  if (!bias_ptr) {
    throw std::runtime_error("bias is null!");
  }
  if (!d0_ptr) {
    throw std::runtime_error("d0_ptr is null!");
  }
{% if has_d1 %}
  if (!d1_ptr) {
    throw std::runtime_error("d1_ptr is null!");
  }
{% endif %}

  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this {{function_name}} specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

# For function declaration codegen.
FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  void*,
{% if has_d1 %}
  void*,
{% endif %}
  void*,
  uint8_t*,
{% if support_split_k %}
    int,
{% endif %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(weight_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
  cudaStream_t
);
"""
)


# For function call codegen.
FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{% if is_profiler %}
{{indent}}    gemm_op,
{% endif %}
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{{indent}}    {{bias_ptr}},
{{indent}}    {{d0_ptr}},
{% if has_d1 %}
{{indent}}    {{d1_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{{indent}}    global_workspace_,
{% if support_split_k %}
{{indent}} {{split_k}},
{% endif %}
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
{{indent}}}
"""
)

# For profiler codegen.
ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
{% if support_split_k %}
  int split_k = std::atoi(argv[4]);
{% endif %}
  {{layout.args_parser}}
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = a_dim0 * a_dim1;
  int64_t b_ptr_sz = b_dim0 * b_dim1;
  int64_t c_ptr_sz = c_dim0 * c_dim1;
  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});
  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz + c_dim1 + c_ptr_sz;
{% if has_d1 %}
  one_copy_sz += c_ptr_sz;
{%endif%}
  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz, device_properties.l2CacheSize);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2
  memory_pool->AllocateTensor(c_dim1, mem_pool_sz);  // bias_ptr: index 3
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz);  // d0 ptr: index 4
{% if has_d1 %}
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz);  // d1 ptr: index 5
{% endif %}
"""
)


def _support_split_k(func_attrs):
    return func_attrs["split_k"] is not None


def _replace_epilogue_cutlass_3x(
    op_def,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
):
    # example of the generated epilogue replaced by this function:
    # ------------------------------------------------------------
    # using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_2x1x1_0_tnt_align8_warpspecialized_pingpong_epi_tma_epilogue =
    # typename cutlass::epilogue::collective::CollectiveBuilder<
    #     cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    #     cute::Shape<cute::_256, cute::_128, cute::_64>,
    #     cute::Shape<cute::_2,cute::_1,cute::_1>,
    #     cutlass::epilogue::collective::EpilogueTileAuto,
    #     float, float,
    #     cutlass::half_t, cutlass::layout::RowMajor, 8,
    #     cutlass::half_t, cutlass::layout::RowMajor, 8,
    #     cutlass::epilogue::TmaWarpSpecialized
    # >::CollectiveOp;

    # The CollectiveBuilder header is a fixed 10-line prefix (indices 0..9):
    #   0 using <name>_epilogue =
    #   1 typename cutlass::epilogue::collective::CollectiveBuilder<
    #   2 cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    #   3 <tile shape>,
    #   4 <cluster shape>,
    #   5 cutlass::epilogue::collective::EpilogueTileAuto,
    #   6 <element_accumulator>, <element_compute>,
    #   7 <element_c>, <layout_c>, <align>,
    #   8 <element_d>, <layout_d>, <align>,
    #   9 <epilogue_schedule>
    # cutlass v4.6.1 then appends an explicit fusion operation (e.g. a multi-line
    # `cutlass::epilogue::fusion::LinearCombination<...>`) before the closing
    # `>::CollectiveOp;`, so the block is no longer exactly 11 lines. Only the
    # fixed prefix indices 0..9 are parsed here (and the whole block is replaced),
    # so key off the actual collected length instead of a hardcoded 11.
    CUTLASS_3X_EPILOGUE_PREFIX_LINES = 10

    lines = op_def.split("\n")
    stripped_lines = [line.strip() for line in lines]
    epilogue_start, epilogue_lines = None, []
    for i in range(len(stripped_lines)):
        if stripped_lines[i].endswith("_epilogue ="):
            epilogue_start = i
            for j in range(i, len(stripped_lines)):
                epilogue_lines.append(stripped_lines[j])
                if stripped_lines[j].endswith("::CollectiveOp;"):
                    break
            break

    if epilogue_start is None:
        raise ValueError(
            f"Generated epilogue not found in the CUTLASS 3.x op_def:\n\n{op_def}"
        )
    if len(epilogue_lines) < CUTLASS_3X_EPILOGUE_PREFIX_LINES + 1:
        raise ValueError(
            f"Generated CUTLASS 3.x epilogue must be at least "
            f"{CUTLASS_3X_EPILOGUE_PREFIX_LINES + 1} lines long, "
            f"but got {len(epilogue_lines)}:\n\n{op_def}"
        )

    epilogue_name = epilogue_lines[0].split(" ")[1]
    element_c, layout_c = epilogue_lines[7].split(", ")[:2]
    element_d, layout_d = epilogue_lines[8].split(", ")[:2]
    element_accumulator, element_compute = epilogue_lines[6].split(",")[:2]
    element_compute = element_compute.strip()
    # v4.6.1 emits the schedule with a trailing comma (a fusion arg follows);
    # the broadcast template appends `>>` after it, so drop any trailing comma.
    epilogue_schedule = epilogue_lines[9].rstrip(",").strip()

    new_epilogue = EPILOGUE_TENSOR_BROADCAST_TEMPLATE.render(
        epilogue_name=epilogue_name,
        element_c=element_c,
        layout_c=layout_c,
        element_d=element_d,
        layout_d=layout_d,
        element_accumulator=element_accumulator,
        element_compute=element_compute,
        unary_op1=unary_op1,
        binary_op1=binary_op1,
        binary_op2=binary_op2,
        unary_op2=unary_op2,
        epilogue_schedule=epilogue_schedule,
        has_d1=(binary_op2 is not None),
    )

    lines_before = lines[:epilogue_start]
    lines_after = lines[epilogue_start + len(epilogue_lines) :]
    new_lines = lines_before + [new_epilogue] + lines_after
    new_op_def = "\n".join(new_lines)

    return new_op_def


def gemm_bias_broadcast_instance(
    op_def,
    func_attrs,
    for_profiler,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
    elem_type,
    cutlass_3x=False,
    sm100_evt=False,
    sm100_mul_header="",
):
    """
    adjust gemm instance with respect to input_accessors, layout and epilogue ops
    """
    if cutlass_3x and sm100_evt:
        # SM100 EVT: op.epilogue_functor (LinCombPerColBias[EltAct] / LinCombPerColBiasMul) was
        # already emitted by the collective builder; leave the instance as-is (no TensorBroadcast).
        # For the mul kind, prepend the (include-guarded) FusionCallbacks header it needs.
        return sm100_mul_header + op_def
    if cutlass_3x:
        return _replace_epilogue_cutlass_3x(
            op_def=op_def,
            unary_op1=unary_op1,
            binary_op1=binary_op1,
            binary_op2=binary_op2,
            unary_op2=unary_op2,
        )

    op_def = common.update_alignments_in_gemm_instance(op_def, func_attrs, for_profiler)
    gemm_universal_params = common.get_gemm_instance_template_params(op_def)
    epilogue_pattern = re.compile(r"\s*(cutlass::epilogue::thread::.*)\s*<")
    match = epilogue_pattern.match(gemm_universal_params[9])
    if match is None:
        raise RuntimeError("Invalid epilogue functor:\n" + gemm_universal_params[9])
    epilogue_functor = match.groups()[0]

    if (
        "use_fp16_acc" in Target.current()._kwargs
        and Target.current()._kwargs["use_fp16_acc"]
        and elem_type == "cutlass::half_t"
    ):
        acc_type = "cutlass::half_t"
    else:
        acc_type = "float"
    gemm_universal_with_broadcast_params = (
        GEMM_UNIVERSAL_WITH_BROADCAST_TEMPLATE.render(
            arch=gemm_universal_params[5],
            tb_shape=gemm_universal_params[6],
            warp_shape=gemm_universal_params[7],
            instruction_shape=gemm_universal_params[8],
            epilogue_functor=epilogue_functor,
            epilogue_vector_length=gemm_universal_params[11],
            unary_op1=unary_op1,
            binary_op1=binary_op1,
            binary_op2=binary_op2,
            unary_op2=unary_op2,
            stage=gemm_universal_params[16],
            alignment_a=gemm_universal_params[17],
            alignment_b=gemm_universal_params[18],
            layout=layout,
            acc_type=acc_type,
            elem_type=elem_type,
            has_d1=(binary_op2 is not None),
        )
    )
    res = re.sub(
        r"cutlass::gemm::device::Gemm<[\s\S]+>;",
        gemm_universal_with_broadcast_params,
        op_def,
    )
    return res


def gemm_bias_broadcast_config(
    func_attrs,
    layout,
    dtype="float16",
    unary_op1=None,
    binary_op1=None,
    binary_op2=None,
    unary_op2=None,
):
    common.make_fproc(
        func_attrs=func_attrs,
        layout=layout,
        include_cutlass_3x_ops=True,
    )
    if Target.current()._arch not in _residual_evt_arches():
        return
    # Route the single-residual add / add_relu configs through the EVT LinCombPerColBias[EltAct]
    # functor (non-transposed); drop every other config's 3.x ops to the SM80 fallback.
    # Always on for SM100; opt-in for SM90 (AIT_SM90_RESIDUAL_EVT) -- see _residual_evt_arches.
    from aitemplate.backend.cuda.gemm_universal import (
        common_bias_activation,
        gemm_rcr_bias,
    )

    import cutlass_lib

    lib = cutlass_lib.library
    kind = _sm100_evt_kind(unary_op1, binary_op1, binary_op2, unary_op2)
    drop = []
    for name, op in func_attrs["op_instance"].items():
        if not common.has_tma_epilogue(op):
            continue  # SM80 2.x op: keep as fallback
        if kind is _UNSUPPORTED_SM100:
            drop.append(name)  # add_add / mul_add (2nd aux tensor) -> SM80 fallback
            continue
        # The residual candidates that survive default_fproc carry StreamK schedulers, dynamic
        # clusters, and explicit TmaWarpSpecialized* schedules -- not all of which build the
        # residual EVT collective. Normalize each to the buildable, profileable form (arch-aware).
        _arch = Target.current()._arch
        _td = getattr(op, "tile_description", None)
        _cluster = list(getattr(_td, "cluster_shape", []) or [])
        if (not _cluster) or _cluster[0] == 0 or _cluster[1] == 0:
            drop.append(name)  # dynamic cluster [0,0,1]: not buildable with the residual EVT
            continue
        if _arch == "100" and _cluster != [1, 1, 1]:
            # SM100: only the single-CTA [1,1,1] cluster builds the residual EVT collective;
            # multi-CTA / oversized clusters fail. Any [1,1,1] tile (64x128/128x128/128x256) is
            # kept so the profiler tunes per shape; all are ~2x SM80 on these memory-bound gemms.
            drop.append(name)
            continue
        # SM90 (opt-in) keeps its standard static clusters ([2,1,1]/[1,2,1]) -- the Sm90 EVT
        # builds with those; an SM100-style [1,1,1]-only filter would drop them all.
        if getattr(op, "tile_scheduler", None) == lib.TileSchedulerType.StreamK:
            op.tile_scheduler = lib.TileSchedulerType.Default
        if _arch == "100":
            # SM100: Auto -> a TMA warp-specialized kernel+epilogue that supports the fusion.
            op.kernel_schedule = lib.KernelScheduleType.ScheduleAuto
            op.epilogue_schedule = lib.EpilogueScheduleType.ScheduleAuto
        else:
            # SM90: Auto -> NoSmemWarpSpecialized ("Auto schedule doesn't support fusion");
            # fusion needs an explicit TmaWarpSpecialized epilogue. Keep the op's own
            # warp-specialized mainloop schedule (pingpong/cooperative) from the generator.
            op.epilogue_schedule = lib.EpilogueScheduleType.TmaWarpSpecialized
        kwargs = dict(
            element_output=lib.DataTypeTag[op.D.element],
            element_compute=lib.DataTypeTag[op.element_epilogue],
            element_bias=lib.DataTypeTag[op.A.element],
            element_source=lib.DataTypeTag[op.A.element],
            element_scalar=lib.DataTypeTag[op.element_epilogue],
        )
        op_kind, activation = kind
        if op_kind == "mul":
            op.epilogue_functor = _LinCombPerColBiasMulFunctor(**kwargs)
        elif activation is None:
            op.epilogue_functor = gemm_rcr_bias._LinCombPerColBiasFunctor(**kwargs)
        else:
            op.epilogue_functor = common_bias_activation._LinCombPerColBiasEltActFunctor(
                activation=activation, **kwargs
            )
    for name in drop:
        del func_attrs["op_instance"][name]


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
):
    import cutlass_lib

    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    op_instance, _ = common.filter_cutlass_3x_ops(op_instance, func_attrs)

    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    support_split_k = _support_split_k(func_attrs)
    has_d1 = common.has_d1(func_attrs)
    _kind = (
        _sm100_evt_kind(unary_op1, binary_op1, binary_op2, unary_op2)
        if Target.current()._arch in _residual_evt_arches()
        else _UNSUPPORTED_SM100
    )
    sm100_evt = _kind is not _UNSUPPORTED_SM100
    sm100_mul_header = SM100_MUL_EVT_HEADER if sm100_evt and _kind[0] == "mul" else ""
    profiler_args_3x = (
        SM100_RESIDUAL_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X
        if sm100_evt
        else PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X
    )

    ndims = 2
    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    instance_name_base = "GemmInstance"
    exec_program = common.EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        problem_args=PROFILER_PROBLEM_ARGS_TEMPLATE.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
            support_split_k=support_split_k,
            layout=layout,
            has_d1=has_d1,
        ),
        problem_args_cutlass_3x=profiler_args_3x.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
            layout=layout,
            has_d1=has_d1,
        ),
    )
    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
    )

    function_name = "gemm"
    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = common.emit_instance(
            op,
            for_profiler=True,
            f_instance_convertor=partial(
                gemm_bias_broadcast_instance,
                layout=layout,
                unary_op1=unary_op1,
                binary_op1=binary_op1,
                binary_op2=binary_op2,
                unary_op2=unary_op2,
                elem_type=elem_input_type,
                sm100_evt=sm100_evt,
                sm100_mul_header=sm100_mul_header,
            ),
        )
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_op_{instance_idx}"
        cutlass_3x = op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x
        instance_template = (
            common.INSTANCE_TEMPLATE_CUTLASS_3X
            if cutlass_3x
            else common.INSTANCE_TEMPLATE
        )
        instance = instance_template.render(
            config_name=common.extract_config_name(
                config,
                cutlass_3x=cutlass_3x,
            ),
            name=instance_name,
            config=config,
        )
        benchmark_instance = common.BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
            support_split_k=support_split_k,
            split_k="split_k",
        )
        instances.append(instance)
        benchmark_instances.append(benchmark_instance)
    op_func = SRC_TEMPLATE.render(
        is_profiler=True,
        instances="\n".join(instances),
        function_name=function_name,
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        input_ndims=ndims,
        weight_ndims=ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_program,
        output_addr_calculator=common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N"
        ),
        support_split_k=support_split_k,
        has_d1=has_d1,
    )
    benchmark_adims = ["a_dim" + str(i) for i in range(ndims)]
    benchmark_bdims = ["b_dim" + str(i) for i in range(ndims)]
    benchmark_cdims = ["c_dim" + str(i) for i in range(ndims)]
    func_call = FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name="gemm",
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b_ptr="memory_pool->RequestTensorByIdx(1)",
        c_ptr="memory_pool->RequestTensorByIdx(2)",
        d0_ptr="memory_pool->RequestTensorByIdx(4)",
        d1_ptr="memory_pool->RequestTensorByIdx(5)",
        bias_ptr="memory_pool->RequestTensorByIdx(3)",
        adims=benchmark_adims,
        bdims=benchmark_bdims,
        cdims=benchmark_cdims,
        support_split_k=support_split_k,
        split_k="split_k",
        has_d1=has_d1,
    )
    code = common.PROFILER_TEMPLATE.render(
        op_func=op_func,
        has_bias=True,
        has_d=True,
        has_d1=has_d1,
        support_split_k=support_split_k,
        args_parse=ARGS_PARSER_TEMPLATE.render(
            layout=layout, support_split_k=support_split_k
        ),
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        func_call=func_call,
        name=instance_name_base,
        tensor_decl=TENSOR_DECL_TEMPLATE.render(has_d1=has_d1),
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return common.build_profiler(file_pairs)


def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    input_addr_calculator = gemm_rcr.get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    support_split_k = _support_split_k(func_attrs)
    has_d1 = common.has_d1(func_attrs)
    _kind = (
        _sm100_evt_kind(unary_op1, binary_op1, binary_op2, unary_op2)
        if Target.current()._arch in _residual_evt_arches()
        else _UNSUPPORTED_SM100
    )
    sm100_evt = _kind is not _UNSUPPORTED_SM100
    sm100_mul_header = SM100_MUL_EVT_HEADER if sm100_evt and _kind[0] == "mul" else ""
    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        layout=layout,
        support_split_k=support_split_k,
        has_d1=has_d1,
    )
    args_3x_template = (
        SM100_RESIDUAL_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X
        if sm100_evt
        else PROBLEM_ARGS_TEMPLATE_CUTLASS_3X
    )
    problem_args_cutlass_3x = args_3x_template.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        layout=layout,
        has_d1=has_d1,
    )
    return common.gen_function(
        func_attrs=func_attrs,
        src_template=SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        f_instance_convertor=partial(
            gemm_bias_broadcast_instance,
            layout=layout,
            unary_op1=unary_op1,
            binary_op1=binary_op1,
            binary_op2=binary_op2,
            unary_op2=unary_op2,
            elem_type=elem_input_type,
            sm100_evt=sm100_evt,
            sm100_mul_header=sm100_mul_header,
        ),
        support_split_k=support_split_k,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N",
            output_accessor=func_attrs["output_accessors"][0],
        ),
    )


def gen_function_decl(func_attrs):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=_support_split_k(func_attrs),
        has_d1=common.has_d1(func_attrs),
    )


def gen_function_call(func_attrs, indent="  "):
    has_d1 = common.has_d1(func_attrs)
    if has_d1:
        (a, b, bias, d0, d1) = func_attrs["inputs"]
    else:
        (a, b, bias, d0) = func_attrs["inputs"]
        d1 = None
    c = func_attrs["outputs"][0]
    # overwrite the global defs if we have input TensorAccessor
    local_dim_defs = common.gen_local_dim_defs(func_attrs, indent=indent)
    adims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][0].original_shapes
    ]
    bdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][1].original_shapes
    ]
    cdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["output_accessors"][0].original_shapes
    ]
    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        bias_ptr=bias._attrs["name"],
        d0_ptr=d0._attrs["name"],
        d1_ptr=d1._attrs["name"] if has_d1 else "",
        c_ptr=c._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
        support_split_k=_support_split_k(func_attrs),
        has_d1=has_d1,
    )
