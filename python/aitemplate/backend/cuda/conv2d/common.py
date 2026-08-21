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
common template for conv2d
"""

import re
from collections import OrderedDict
from hashlib import sha1
from typing import List

import jinja2
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal.common import add_profiler, build_profiler
from aitemplate.backend.target import Target
from aitemplate.utils import alignment


KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass{{opcode_class}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = cutlass::conv::device::ImplicitGemmConvolution<{{config_name}}>;
"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}using ElementComputeEpilogue = typename {{instance_name}}::ElementCompute;
{{indent}}//  TODO: cast to right dtype
{{indent}}typename {{instance_name}}::Arguments arguments{
{{indent}}    problem_size,                                                                 // ConvProblemSize const & problem_size
{{indent}}    {static_cast<{{dtype}}*>(in_ptr), layout_A},                                  // TensorRefA const & ref_A
{{indent}}    {static_cast<{{dtype}}*>(weight_ptr), layout_B},                              // TensorRefA const & ref_B
{% if is_bias %}
{{indent}}    {static_cast<{{dtype}}*>(bias_ptr), cutlass::layout::TensorNHWC::Stride(0)},  // TensorRefC const & ref_C
{% elif is_bias_add %}
{{indent}}    {static_cast<{{dtype}}*>(res_ptr), layout_C},                                 // TensorRefC const & ref_C
{% else %}
{{indent}}    {static_cast<{{dtype}}*>(out_ptr), layout_C},                                 // TensorRefC const & ref_C
{% endif %}
{{indent}}    {static_cast<{{dtype}}*>(out_ptr), layout_C},                                 // TensorRefC const & ref_D
{% if is_bias %}
{{indent}}    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},                       // typename EpilogueOutputOp::Params const & output_op
{% elif is_bias_add %}
{{indent}}    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},                       // typename EpilogueOutputOp::Params const & output_op
{{indent}}    cutlass::conv::SplitKMode::kSerial,                                           // SplitKMode const & split_k_mode
{{indent}}    static_cast<{{dtype}}*>(bias_ptr),                                            // void * ptr_Vector
{{indent}}    nullptr,                                                                      // void * ptr_Tensor
{{indent}}    0,                                                                            // typename LayoutC::Stride::Index ldr
{{indent}}    *out_ch,                                                                      // typename LayoutC::Stride::Index ldt
{% else %}
{{indent}}    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},                       // typename EpilogueOutputOp::Params const & output_op
{% endif %}
{{indent}}};
{{indent}}{{instance_name}} conv_op;
{% if is_profiler %}
{{indent}}size_t workspace_size = conv_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE_{{instance_name}} = workspace_size;
{% endif %}
{{indent}}auto status = conv_op.can_implement(arguments);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = conv_op.initialize(arguments, workspace);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = conv_op(stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <cstdio>
#include <stdexcept>

#include "cutlass/cutlass.h"
{% if is_transpose %}
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
{% elif is_depthwise %}
#include "cutlass/conv/kernel/default_depthwise_fprop.h"
{% else %}
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_group_fprop.h"
{% endif %}
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

{{extra_header}}

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      static char msg[2048];                                                          \\
      snprintf(msg, sizeof(msg), "[%s] Got cutlass error: %s at: %s",                 \\
        __FILE__, cutlassGetStatusString(error), __LINE__);                           \\
      fprintf(stderr, msg);                                                           \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}

{{functions}}
"""
)

FUNCTION_TEMPLATE = jinja2.Template(
    """
void {{function_name}} (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,
{% if is_bias %}
    void* bias_ptr,
{% elif is_bias_add %}
    void* bias_ptr,
    void* res_ptr,
{% endif %}
    uint8_t* workspace,
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int strideh,
    int dilationh,
    int padh,
    int stridew,
    int dilationw,
    int padw,
    cudaStream_t stream
  ) {

  {{shape_function}}

  int i32_batch = *batch;
  int i32_in_h = *in_h;
  int i32_in_w = *in_w;
  int i32_in_ch = *in_ch;
  int i32_out_ch = *out_ch;
  int i32_kernel_h = *kernel_h;
  int i32_kernel_w = *kernel_w;
  int i32_out_batch = *out_batch;
  int i32_out_h = *out_h;
  int i32_out_w = *out_w;

  using cutlass::layout::TensorNHWC;
  TensorNHWC layout_A(TensorNHWC::packed(cutlass::make_Coord(i32_batch, i32_in_h, i32_in_w, i32_in_ch)));
{% if is_depthwise%}
  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_out_ch, i32_kernel_h, i32_kernel_w, 1)));
{% elif is_transpose %}
  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_in_ch, i32_kernel_h, i32_kernel_w, i32_out_ch)));
{% else %}
  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch)));
{% endif %}
  TensorNHWC layout_C(TensorNHWC::packed(cutlass::make_Coord(i32_out_batch, i32_out_h, i32_out_w, i32_out_ch)));

  cutlass::conv::Conv2dProblemSize problem_size(
{% if is_transpose %}
    {i32_out_batch, i32_out_h, i32_out_w, i32_out_ch},    // cutlass::Tensor4DCoord input_size
{% else %}
    {i32_batch, i32_in_h, i32_in_w, i32_in_ch},           // cutlass::Tensor4DCoord input_size
{% endif %}
{% if is_depthwise%}
    {i32_out_ch, i32_kernel_h, i32_kernel_w, 1},  // cutlass::Tensor4DCoord filter_size
{% elif is_transpose%}
    {i32_in_ch, i32_kernel_h, i32_kernel_w, i32_out_ch},  // cutlass::Tensor4DCoord filter_size
{% else %}
    {i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch},  // cutlass::Tensor4DCoord filter_size
{% endif %}
    {padh, padh, padw, padw},                                 // cutlass::Tensor4DCoord padding
    {strideh, stridew},                                     // cutlass::MatrixCoord stride
    {dilationh, dilationw},                                 // cutlass::MatrixCoord dilation
{% if is_transpose %}
    {i32_batch, i32_in_h, i32_in_w, i32_in_ch},           // cutlass::Tensor4DCoord output_size
{% else %}
    {i32_out_batch, i32_out_h, i32_out_w, i32_out_ch},    // cutlass::Tensor4DCoord output_size
{% endif %}
    cutlass::conv::Mode::kCrossCorrelation,               // cutlass::conv::Mode mode
    1                                                     // int split_k_slices
  );

  {{exec_paths}}

  throw std::runtime_error(
    "Unsupported workload for this conv2d specialization."
  );
}
"""
)

BENCHMARK_INSTANCE_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  int ret = 0;
{{indent}}  try {
{{indent}}    ret = {{func_name}}(
{{indent}}      &runtime,
{{indent}}      &workspace_size,
{{indent}}      {{ni}},
{{indent}}      {{hi}},
{{indent}}      {{wi}},
{{indent}}      {{ci}},
{{indent}}      {{co}},
{{indent}}      {{kh}},
{{indent}}      {{kw}},
{{indent}}      {{no}},
{{indent}}      {{ho}},
{{indent}}      {{wo}},
{{indent}}      {{strideh}},
{{indent}}      {{dilationh}},
{{indent}}      {{padh}},
{{indent}}      {{stridew}},
{{indent}}      {{dilationw}},
{{indent}}      {{padw}},
{{indent}}      global_workspace_,
{{indent}}      stream
{{indent}}    );
{{indent}}  } catch (...) {
{{indent}}    runtime = 0;
{{indent}}    workspace_size = 0;
{{indent}}  }
{{indent}}  if (ret != 0)
{{indent}}    return ret;
{{indent}}  std::cout << "OP:{{conv_op_name}},"
{{indent}}            << "TIME:" << runtime << ","
{{indent}}            << "WS:" << workspace_size << std::endl;
{{indent}}}
"""
)

BENCHMARK_DECL_TEMPLATE = jinja2.Template(
    """
int benchmark_{{function_name}} (
  float*,
  size_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int,
  int,
  int,
  int,
  int,
  int,
  uint8_t*,
  cudaStream_t
);
"""
)

BENCHMARK_TEMPLATE = jinja2.Template(
    """
int benchmark_{{function_name}} (
  float* runtime,
  size_t* workspace_size,
  int64_t NI,
  int64_t HI,
  int64_t WI,
  int64_t CI,
  int64_t CO,
  int64_t KH,
  int64_t KW,
  int64_t NO,
  int64_t HO,
  int64_t WO,
  int strideh,
  int dilationh,
  int padh,
  int stridew,
  int dilationw,
  int padw,
  uint8_t* global_workspace_,
  cudaStream_t stream
) {
  using ElementInputA = typename {{instance_name}}::ElementA;
  using ElementInputB = typename {{instance_name}}::ElementB;
  using ElementOutput = typename {{instance_name}}::ElementC;

  cutlass::HostTensor<ElementInputA, typename {{instance_name}}::LayoutA> x({NI, HI, WI, CI});
  cutlass::HostTensor<ElementInputB, typename {{instance_name}}::LayoutB> w({CO, KH, KW, CI});
{% if is_bias %}
  cutlass::HostTensor<ElementInputB, typename {{instance_name}}::LayoutB> b({(int)CO, 1, 1, 1});
{% elif is_bias_add %}
  cutlass::HostTensor<ElementInputB, typename {{instance_name}}::LayoutB> b({(int)CO, 1, 1, 1});
  cutlass::HostTensor<ElementOutput, typename {{instance_name}}::LayoutC> r({NO, HO, WO, CO});
{% endif %}
  cutlass::HostTensor<ElementOutput, typename {{instance_name}}::LayoutC> y({NO, HO, WO, CO});

  // warmup
{{func_call}}
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 5; ++i) {
{{func_call}}
  }
  cudaEventRecord(events[1], stream);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  *runtime = runtime_ms;
  *workspace_size = GLOBAL_WORKSPACE_SIZE_{{instance_name}};
  return 0;
}
"""
)

PROFILER_BENCHMARK_TEMPLATE = jinja2.Template(
    """
static size_t GLOBAL_WORKSPACE_SIZE_{{instance_name}} = 0;

{{op_source}}

{{benchmark}}
"""
)

PROFILER_MAIN_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"

{{benchmark_decls}}

int main(int argc, char** argv) {
  int64_t batch = std::stoi(argv[1]);
  int64_t in_h = std::stoi(argv[2]);
  int64_t in_w = std::stoi(argv[3]);
  int64_t in_ch = std::stoi(argv[4]);
  int64_t kernel_h = std::stoi(argv[5]);
  int64_t kernel_w = std::stoi(argv[6]);
  int64_t out_ch = std::stoi(argv[7]);
  int strideh = std::stoi(argv[8]);
  int padh = std::stoi(argv[9]);
  int dilationh = std::stoi(argv[10]);
  int stridew = std::stoi(argv[11]);
  int padw = std::stoi(argv[12]);
  int dilationw = std::stoi(argv[13]);

{{shape_func}}

  float runtime = 0;
  size_t workspace_size = 0;
  uint8_t* global_workspace_ = nullptr;
  cudaStream_t stream = nullptr;

{{benchmark_instances}}

  return 0;
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
{% if is_bias %}
  void*,
{% elif is_bias_add %}
  void*,
  void*,
{% endif %}
  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  cudaStream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{weight_ptr}},
{{indent}}    {{out_ptr}},
{% if is_bias %}
{{indent}}    {{bias_ptr}},
{% elif is_bias_add %}
{{indent}}    {{bias_ptr}},
{{indent}}    {{res_ptr}},
{% endif %}
{{indent}}    global_workspace_,
{{indent}}    {{p_batch}},
{{indent}}    {{p_out_ch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_kernel_h}},
{{indent}}    {{p_kernel_w}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{strideh}},
{{indent}}    {{dilationh}},
{{indent}}    {{padh}},
{{indent}}    {{stridew}},
{{indent}}    {{dilationw}},
{{indent}}    {{padw}},
{{indent}}    stream
{{indent}});
"""
)


# ---------------------------------------------------------------------------
# CUTLASS 3.x (SM90 / Hopper) native convolution host codegen.
#
# AIT's default conv host path (templates above) is 100% CUTLASS 2.x: it wraps
# the kernel in cutlass::conv::device::ImplicitGemmConvolution and builds the
# SM80 positional Arguments. The templates below are a parallel host path for
# ConvOperation3x (is_3x=True) ops: a ConvUniversalAdapter<ConvUniversal<...>>
# with a hand-authored fusion epilogue (bias / bias+relu / bias+add+relu),
# mirroring test/unit/conv/device_3x/testbed_conv.hpp and
# examples/76_blackwell_conv. Gated entirely on op.is_3x so the 2.x (FORCE=0)
# path is byte-identical.
# ---------------------------------------------------------------------------

# EmitConv3xInstance (3rdparty conv3x_emitter.py) leaves the epilogue
# CollectiveBuilder's fusion (15th) template arg defaulted to a plain
# LinearCombination -- so it can never thread bias/residual/relu. This template
# is that emitter's output with ${fusion_op} injected as the fusion arg.
INSTANCE_TEMPLATE_3X_STR = """

// CUTLASS >= 3 convolution ${conv_kind_name} kernel instance "${operation_name}"
using ${operation_name}_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch},
    ${opcode_class_epi},
    ${mma_tile_shape},               // mma tile shape
    ${cluster_shape},                // cluster shape
    ${epi_tile_mn},
    ${element_accumulator},
    ${element_compute},
    ${element_c}, ${layout_c}, 128 / cute::sizeof_bits_v<${element_c}>,
    ${element_d}, ${layout_d}, 128 / cute::sizeof_bits_v<${element_d}>,
    ${epilogue_schedule},
    ${fusion_op}
  >::CollectiveOp;

using ${operation_name}_mainloop =
  typename cutlass::conv::collective::CollectiveBuilder<
    ${arch},
    ${opcode_class_main},
    ${conv_kind},         // kFprop, kDgrad, or kWgrad
    ${element_a}, ${layout_a}, 128 / cute::sizeof_bits_v<${element_a}>,
    ${element_b}, ${layout_b}, 128 / cute::sizeof_bits_v<${element_b}>,
    ${element_accumulator},
    ${mma_tile_shape},        // mma tile shape
    ${cluster_shape},         // cluster shape
    ${stages},
    ${kernel_schedule}
  >::CollectiveOp;

using ${operation_name}_problem_shape = cutlass::conv::ConvProblemShape<${conv_kind}, ${operation_name}_mainloop::NumSpatialDimensions>;

using ${operation_name}_base = cutlass::conv::kernel::ConvUniversal<
    ${operation_name}_problem_shape,
    ${operation_name}_mainloop,
    ${operation_name}_epilogue,
    ${tile_scheduler}
  >;
"""

INSTANCE_TEMPLATE_3X = jinja2.Template(
    """
{{config}}
using {{name}} = cutlass::conv::device::ConvUniversalAdapter<{{config_name}}>;
"""
)

SRC_TEMPLATE_3X = jinja2.Template(
    """
#include <cstdio>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/device_memory.h"

{{extra_header}}

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      static char msg[2048];                                                          \\
      snprintf(msg, sizeof(msg), "[%s] Got cutlass error: %s at: %s",                 \\
        __FILE__, cutlassGetStatusString(error), __LINE__);                           \\
      fprintf(stderr, msg);                                                           \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}

{{functions}}
"""
)

FUNCTION_TEMPLATE_3X = jinja2.Template(
    """
void {{function_name}} (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,
{% if is_bias %}
    void* bias_ptr,
{% elif is_bias_add %}
    void* bias_ptr,
    void* res_ptr,
{% endif %}
    uint8_t* workspace,
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int strideh,
    int dilationh,
    int padh,
    int stridew,
    int dilationw,
    int padw,
    cudaStream_t stream
  ) {

  {{shape_function}}

  int i32_batch = *batch;
  int i32_in_h = *in_h;
  int i32_in_w = *in_w;
  int i32_in_ch = *in_ch;
  int i32_out_ch = *out_ch;
  int i32_kernel_h = *kernel_h;
  int i32_kernel_w = *kernel_w;
  int i32_out_batch = *out_batch;
  int i32_out_h = *out_h;
  int i32_out_w = *out_w;
  (void)i32_out_batch; (void)i32_out_h; (void)i32_out_w;

  using ConvProblemShape3x = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 2>;
  ConvProblemShape3x problem_shape(
    cutlass::conv::Mode::kCrossCorrelation,
    {i32_batch, i32_in_h, i32_in_w, i32_in_ch},           // [n, h, w, c]
    {i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch},  // [k, r, s, c]
    {padh, padw},                                         // lower padding [pad_h, pad_w]
    {padh, padw},                                         // upper padding [pad_h, pad_w]
    {strideh, stridew},                                   // traversal stride [stride_h, stride_w]
    {dilationh, dilationw},                               // dilation [dilation_h, dilation_w]
    1                                                     // groups
  );

  {{exec_paths}}

  throw std::runtime_error(
    "Unsupported workload for this conv2d specialization."
  );
}
"""
)

# EpilogueArguments: {fusion_thread_args, ptr_C, stride_C, ptr_D, stride_D}.
# StrideC/StrideD for fprop are derived from problem_shape.stride_C exactly as
# testbed_conv.hpp does. The fusion callback args (alpha/beta/bias_ptr) are set
# by name after aggregate-init so we never depend on brace field order.
EXEC_TEMPLATE_3X = jinja2.Template(
    """
{{indent}}using {{instance_name}}_StrideC = typename {{instance_name}}::ConvKernel::StrideC;
{{indent}}using {{instance_name}}_StrideD = typename {{instance_name}}::ConvKernel::StrideD;
{{indent}}{{instance_name}}_StrideC stride_C{};
{{indent}}{{instance_name}}_StrideD stride_D{};
{{indent}}cute::for_each(cute::make_seq<cute::rank<0>({{instance_name}}_StrideC{})>{}, [&](auto i) {
{{indent}}  cute::get<0, i>(stride_C) = problem_shape.stride_C[ConvProblemShape3x::RankT - 2 - i];
{{indent}}});
{{indent}}cute::for_each(cute::make_seq<cute::rank<0>({{instance_name}}_StrideD{})>{}, [&](auto i) {
{{indent}}  cute::get<0, i>(stride_D) = problem_shape.stride_C[ConvProblemShape3x::RankT - 2 - i];
{{indent}}});
{{indent}}using {{instance_name}}_ElemA = typename {{instance_name}}::ElementA;
{{indent}}using {{instance_name}}_ElemB = typename {{instance_name}}::ElementB;
{{indent}}using {{instance_name}}_ElemC = typename {{instance_name}}::ElementC;
{{indent}}using {{instance_name}}_ElemD = typename {{instance_name}}::ElementD;
{{indent}}typename {{instance_name}}::Arguments arguments{
{{indent}}    problem_shape,
{{indent}}    { static_cast<const {{instance_name}}_ElemA*>(in_ptr), static_cast<const {{instance_name}}_ElemB*>(weight_ptr) },
{{indent}}    {
{{indent}}      {},                                                    // fusion thread args (set below)
{% if is_bias_add %}
{{indent}}      static_cast<const {{instance_name}}_ElemC*>(res_ptr),  // ptr_C (residual)
{% else %}
{{indent}}      nullptr,                                               // ptr_C (unused)
{% endif %}
{{indent}}      stride_C,
{{indent}}      static_cast<{{instance_name}}_ElemD*>(out_ptr),        // ptr_D
{{indent}}      stride_D
{{indent}}    }
{{indent}}};
{{indent}}// alpha descales the f32 accumulator. fp16 conv: alpha=1. fp8 conv: alpha =
{{indent}}// act_scale_inv * w_scale_inv (per-tensor dequant) applied before bias/relu/add.
{{indent}}arguments.epilogue.thread.alpha = {{alpha_expr}};
{% if is_bias_add %}
{{indent}}// beta scales the residual source C. fp16 residual: beta=1. fp8 (full-fusion)
{{indent}}// residual: beta = out_scale/residual_scale (dequant the E4M3 residual, requant).
{{indent}}arguments.epilogue.thread.beta = {{beta_expr}};
{% else %}
{{indent}}arguments.epilogue.thread.beta = 0.0f;
{% endif %}
{% if is_bias or is_bias_add %}
{{indent}}arguments.epilogue.thread.bias_ptr = static_cast<const {{bias_elem}}*>(bias_ptr);
{% endif %}
{{indent}}{{instance_name}} conv_op;
{% if is_profiler %}
{{indent}}size_t workspace_size = conv_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE_{{instance_name}} = workspace_size;
{% endif %}
{{indent}}auto status = conv_op.can_implement(arguments);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = conv_op.initialize(arguments, workspace, stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = conv_op(stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}return;
"""
)

BENCHMARK_TEMPLATE_3X = jinja2.Template(
    """
int benchmark_{{function_name}} (
  float* runtime,
  size_t* workspace_size,
  int64_t NI,
  int64_t HI,
  int64_t WI,
  int64_t CI,
  int64_t CO,
  int64_t KH,
  int64_t KW,
  int64_t NO,
  int64_t HO,
  int64_t WO,
  int strideh,
  int dilationh,
  int padh,
  int stridew,
  int dilationw,
  int padw,
  uint8_t* global_workspace_,
  cudaStream_t stream
) {
  using ElementInputA = typename {{instance_name}}::ElementA;
  using ElementInputB = typename {{instance_name}}::ElementB;
  using ElementOutput = typename {{instance_name}}::ElementD;

  cutlass::HostTensor<ElementInputA, cutlass::layout::TensorNHWC> x({NI, HI, WI, CI});
  cutlass::HostTensor<ElementInputB, cutlass::layout::TensorNHWC> w({CO, KH, KW, CI});
{% if is_bias %}
  cutlass::HostTensor<ElementInputA, cutlass::layout::TensorNHWC> b({(int)CO, 1, 1, 1});
{% elif is_bias_add %}
  cutlass::HostTensor<ElementInputA, cutlass::layout::TensorNHWC> b({(int)CO, 1, 1, 1});
  cutlass::HostTensor<ElementOutput, cutlass::layout::TensorNHWC> r({NO, HO, WO, CO});
{% endif %}
  cutlass::HostTensor<ElementOutput, cutlass::layout::TensorNHWC> y({NO, HO, WO, CO});

  // warmup
{{func_call}}
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 5; ++i) {
{{func_call}}
  }
  cudaEventRecord(events[1], stream);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  *runtime = runtime_ms;
  *workspace_size = GLOBAL_WORKSPACE_SIZE_{{instance_name}};
  return 0;
}
"""
)


def make_fusion_cpp(op, epilogue_name):
    """Select the SM90 fusion epilogue op (and residual flag) for a conv op.

    Maps AIT's conv epilogue name to a CUTLASS 3.x fusion operation. Bias is
    per-output-channel = per-K = the implicit-gemm column dim -> PerCol. All
    three use fusion::LinCombPerColBiasEltAct, which computes
        D = activation(alpha*acc + beta*C + per-col bias)
    (sm90_callbacks_tma_warpspecialized.hpp:823-858). The residual (add) is the
    full-tensor epilogue source C with beta=1, so the add happens *inside* the
    activation -- matching the 2.x LinearCombinationResidualBlock, whose formula
    is UnaryOp(BinaryOp(ActivationOp(acc+bias), residual)) = ReLu(acc+bias+res)
    (linear_combination_residual_block.h:49). Using PerColResAddPerColBiasEltAct
    here would instead give res + ReLu(acc+bias) (add *outside* the activation),
    which is numerically different.
      LinearCombination              -> Identity, no source (bias only)
      LinearCombinationRelu          -> ReLu,     no source
      LinearCombinationResidualBlock -> <unary_op>, source C = residual, beta=1
    """
    from cutlass_lib import library

    elem_out = library.DataTypeTag[op.D.element]
    elem_compute = library.DataTypeTag[op.element_compute]
    # Bias element: normally = C (source) element. But under FULL fp8 fusion the source C
    # is the E4M3 residual, while the per-channel bias must stay f16 for precision -- the
    # fusion op takes a SEPARATE ElementBias, so decouple them: bias is f16 whenever C is
    # e4m3. (The epilogue's source-C element is op.C.element, set independently.)
    if op.C.element == library.DataType.e4m3:
        elem_bias = library.DataTypeTag[library.DataType.f16]
    else:
        elem_bias = library.DataTypeTag[op.C.element]

    def _fusion(act):
        return (
            "cutlass::epilogue::fusion::LinCombPerColBiasEltAct<"
            f"{act}, {elem_out}, {elem_compute}, {elem_bias}>"
        )

    if epilogue_name == "LinearCombinationResidualBlock":
        # 2.x residual block: outer UnaryOp is the activation applied to
        # (acc + bias + residual); inner ActivationOp is Identity here.
        act = library.EpilogueMathTag[op.unary_op]
        return _fusion(act), True
    if epilogue_name == "LinearCombinationRelu":
        return _fusion("cutlass::epilogue::thread::ReLu"), False
    # default: plain bias (LinearCombination)
    return _fusion("cutlass::epilogue::thread::Identity"), False


def emit_instance_3x(op):
    """Emit a CUTLASS 3.x (SM90) conv instance with a fused epilogue.

    Mirrors cutlass_library.conv3x_emitter.EmitConv3xInstance.emit (reusing its
    shape/schedule helpers), but injects op._ait_fusion_cpp as the epilogue
    CollectiveBuilder's fusion op.
    """
    from string import Template

    from cutlass_lib import conv3x_emitter, library

    e = conv3x_emitter.EmitConv3xInstance()

    tile_shape = op.tile_description.tile_shape
    # SM90 (arch < 100): no cta/cluster division, tile == cta shape.
    cta_m, cta_n, cta_k = tile_shape

    opcode_class_main = library.OpcodeClassTag[
        op.tile_description.math_instruction.opcode_class
    ]
    kernel_schedule = library.KernelScheduleTag[op.kernel_schedule].replace(
        "gemm::", "conv::"
    )
    values = {
        "operation_name": op.procedural_name(),
        "conv_kind": library.ConvKindTag[op.conv_kind],
        "conv_kind_name": library.ConvKindNames[op.conv_kind].capitalize(),
        "element_a": library.DataTypeTag[op.A.element],
        "layout_a": library.LayoutTag[op.A.layout],
        "element_b": library.DataTypeTag[op.B.element],
        "layout_b": library.LayoutTag[op.B.layout],
        "element_c": library.DataTypeTag[op.C.element],
        "layout_c": library.LayoutTag[op.C.layout],
        "element_d": library.DataTypeTag[op.D.element],
        "layout_d": library.LayoutTag[op.D.layout],
        "element_accumulator": library.DataTypeTag[op.accumulator_type()],
        "arch": e.arch_number_to_type(op.arch),
        "mma_tile_shape": e.mma_tile_shape(op, cta_m, cta_n, cta_k),
        "cluster_shape": e.cluster_shape(op),
        "opcode_class_epi": opcode_class_main,
        "opcode_class_main": opcode_class_main,
        "epi_tile_mn": "cutlass::epilogue::collective::EpilogueTileAuto",
        "stages": e.stage_count(op),
        "kernel_schedule": kernel_schedule,
        "epilogue_schedule": library.EpilogueScheduleTag[op.epilogue_schedule],
        "tile_scheduler": library.TileSchedulerTag[op.tile_scheduler],
        "element_compute": library.DataTypeTag[op.element_compute],
        "fusion_op": op._ait_fusion_cpp,
    }
    return Template(INSTANCE_TEMPLATE_3X_STR).substitute(values)


def kernel_name(op, layout=None):
    """generate cuda kernel name"""
    from cutlass_lib import library

    threadblock = op.tile_description.procedural_name()
    extended_name = op.extended_name()
    opcode_class_name = library.OpcodeClassNames[
        op.tile_description.math_instruction.opcode_class
    ]
    if layout is None:
        # ConvOperation3x (SM90) does not define layout_name(); fall back to the same
        # short A-layout name the SM80 Conv2dOperation.layout_name() would produce.
        if hasattr(op, "layout_name"):
            layout = op.layout_name()
        else:
            layout = library.ShortLayoutTypeNames[op.A.layout]
    align_ab = op.A.alignment
    align_c = op.C.alignment
    name = KERNEL_KEY_TEMPLATE.render(
        threadblock=threadblock,
        extended_name=extended_name,
        opcode_class_name=opcode_class_name,
        layout=layout,
        align_ab=align_ab,
        align_c=align_c,
    )
    return name.replace("\n", "")


def emit_instance(op):
    """emit instance"""
    import cutlass_lib

    # CUTLASS 3.x (SM90) conv ops take a fully separate host path. Check this
    # first: extract_config may set .binary_op on a 3x op (for residual epilogue
    # bookkeeping), which must NOT route it to the 2.x WithBroadcast emitter.
    if getattr(op, "is_3x", False):
        return emit_instance_3x(op)

    if hasattr(op, "binary_op"):
        emiter = cutlass_lib.conv2d_operation.EmitConv2dWithBroadcastInstance()
    else:
        emiter = cutlass_lib.conv2d_operation.EmitConv2dInstance()
    op_def = emiter.emit(op)
    return op_def


def extract_config(
    func_attrs,
    dtype="float16",
    skip_simt_kernels=False,
    f_apply_special_config=None,
    op_kind=None,
    op_layout=None,
):
    """Extracts cutlass config for conv kernels."""
    import copy

    import cutlass_lib

    spec = CUDASpec()

    # fp8 (E4M3) SM90 3.x conv: A/B are e4m3, C (bias/residual) stays f16, D (output) is
    # f16 by default OR e4m3 when the conv fuses its output into the next conv (producer-
    # epilogue fusion). `ab_type` is the operand element; `c_type`/`d_type` are C/D.
    # d_type is read from the conv's OUTPUT tensor dtype (e4m3 -> fused). For every non-fp8
    # dtype ab_type == c_type == d_type (the historical behavior).
    if dtype == "float8_e4m3":
        data_type = cutlass_lib.library.DataType.e4m3
        ab_type = cutlass_lib.library.DataType.e4m3
        # C is the epilogue source: for a bias_add (residual) conv it's the residual tensor
        # (inputs[3]) -- e4m3 under full fusion, else f16; for bias/bias_relu there is no
        # source so C is just the bias element (f16). D is the output tensor dtype.
        _e4m3 = cutlass_lib.library.DataType.e4m3
        _f16 = cutlass_lib.library.DataType.f16
        _ins = func_attrs["inputs"]
        res_dtype = _ins[3]._attrs["dtype"] if len(_ins) >= 4 else "float16"
        c_type = _e4m3 if res_dtype == "float8_e4m3" else _f16
        out_dtype = func_attrs["outputs"][0]._attrs["dtype"]
        d_type = _e4m3 if out_dtype == "float8_e4m3" else _f16
        acc_type = cutlass_lib.library.DataType.f32
    else:
        lib_dtype = spec.dtype_to_lib_type(dtype)
        if lib_dtype == "float":
            data_type = cutlass_lib.library.DataType.f32
            acc_type = cutlass_lib.library.DataType.f32
        elif "half" in lib_dtype:
            data_type = cutlass_lib.library.DataType.f16
            acc_type = cutlass_lib.library.DataType.f32
            # check target use fp16 acc
            if "use_fp16_acc" in Target.current()._kwargs:
                if Target.current()._kwargs["use_fp16_acc"]:
                    acc_type = cutlass_lib.library.DataType.f16
        elif "bfloat16" in lib_dtype:
            data_type = cutlass_lib.library.DataType.bf16
            acc_type = cutlass_lib.library.DataType.f32
        else:
            raise RuntimeError(f"Unsupported dtype {lib_dtype}")
        ab_type = data_type
        c_type = data_type
        d_type = data_type

    def f_proc_op(op):
        ret = []
        if (
            skip_simt_kernels
            and op.tile_description.math_instruction.opcode_class
            == cutlass_lib.library.OpcodeClass.Simt
        ):
            return ret

        # CUTLASS 3.x conv ops (ConvOperation3x, is_3x=True) are Hopper (SM90) kernels
        # that carry a different attribute set than the SM80 Conv2dOperation. In
        # particular they have no `iterator_algorithm` (that is an SM80-only concept)
        # and they select the epilogue via epilogue_schedule/kernel_schedule rather
        # than `epilogue_functor`/`element_epilogue`. Gate those SM80-only accesses on
        # the op version, mirroring how the gemm path keys off GemmKind.Universal3x.
        is_3x = getattr(op, "is_3x", False)

        # CUTLASS 3.x (SM90 / Hopper) native conv path. These ConvOperation3x ops
        # carry no iterator_algorithm/epilogue_functor; the fused epilogue is
        # hand-authored (see emit_instance_3x / make_fusion_cpp) from the AIT
        # epilogue name. Only the TMA warp-specialized align-8 f16 kernels are
        # realizable, so we keep the op's native alignment (8) rather than
        # expanding low-alignment variants that would not compile.
        if is_3x:
            # fp8: A/B == e4m3, C == f16, D == e4m3 (fused) or f16. f16/bf16/f32:
            # ab_type == c_type == d_type so this is the original all-equal check.
            if not (
                op.A.element == ab_type
                and op.B.element == ab_type
                and op.C.element == c_type
                and op.D.element == d_type
                and op.accumulator_type() == acc_type
            ):
                return ret
            op = copy.deepcopy(op)
            epilogue_name = func_attrs["epilogue"]
            # apply special config if required (sets activation/binary/unary_op
            # on residual ops; harmless bookkeeping for the 3x emitter).
            if f_apply_special_config is not None:
                op = f_apply_special_config(func_attrs, op)
            fusion_cpp, is_residual = make_fusion_cpp(op, epilogue_name)
            op._ait_fusion_cpp = fusion_cpp
            op._ait_is_residual = is_residual
            ret.append(op)
            return ret

        if (
            op.A.element == data_type
            and op.B.element == data_type
            and op.C.element == data_type
            and op.iterator_algorithm
            == cutlass_lib.library.IteratorAlgorithm.Optimized
            and op.tile_description.math_instruction.element_accumulator == acc_type
        ):
            op = copy.deepcopy(op)

            # set epilogue
            epilogue_name = func_attrs["epilogue"]
            op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[
                epilogue_name
            ]
            op.element_epilogue = acc_type

            # apply special config if required
            if f_apply_special_config is not None:
                op = f_apply_special_config(func_attrs, op)

            # set C alignment depending on the dtype
            for i in alignment.get_alignments(dtype):
                op = copy.deepcopy(op)
                op.C.alignment = i
                ret.append(op)

        return ret

    if op_kind is None:
        op_kind = cutlass_lib.library.OperationKind.Conv2d
    extract_ops = list(Target.current()._operators[op_kind].items())
    conv_kind = cutlass_lib.library.ConvKind.Fprop

    conv_ops = OrderedDict()
    for _, value in extract_ops:
        op = value[0]
        if op.conv_kind == conv_kind:
            ret = f_proc_op(op)
            if len(ret) > 0:
                for op_inst in ret:
                    key = kernel_name(op_inst, layout=op_layout)
                    conv_ops[key] = op_inst
    return conv_ops


def _op_a_is_e4m3(op):
    """True if this (3x) conv op's A operand is E4M3 (i.e. an fp8 conv op)."""
    try:
        from cutlass_lib import library

        return getattr(op, "is_3x", False) and op.A.element == library.DataType.e4m3
    except Exception:
        return False


def _conv_lib_dtype(backend_spec, dtype):
    """Map an AIT dtype to its cutlass element type for conv codegen.

    CUDASpec.dtype_to_lib_type has no float8_e4m3 entry (it would raise). The
    SM90 3.x conv exec/emit path no longer uses this scalar dtype for fp8 (each
    pointer is cast to its instance's own ElementA/B/C/D), but the string is
    still threaded for the 2.x path and profiler, so return the cutlass fp8 type
    rather than crashing.
    """
    if dtype == "float8_e4m3":
        return "cutlass::float_e4m3_t"
    return backend_spec.dtype_to_lib_type(dtype)


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
    f_emit_instance=emit_instance,
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
    instance_name_base="DeviceConvFwdInstance",
):
    """Generate profiler sources."""
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]

    backend_spec = CUDASpec()
    dtype = _conv_lib_dtype(backend_spec, func_attrs["inputs"][0]._attrs["dtype"])

    func_call_extra_args = {}
    if is_bias:
        func_call_extra_args = {
            "bias_ptr": "b.device_data()",
        }
    elif is_bias_add:
        func_call_extra_args = {
            "bias_ptr": "b.device_data()",
            "res_ptr": "r.device_data()",
        }

    benchmark_decls = []
    benchmark_instances = []
    profiler_benchmarks = {}

    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        is_3x = getattr(op, "is_3x", False)
        config = f_emit_instance(op)
        instance_name = f"{instance_name_base}_{instance_idx}"
        function_name = f"{op_type}_{op_name}"

        if is_3x:
            config_name = op.procedural_name() + "_base"
            _op_is_fp8 = _op_a_is_e4m3(op)
            exec_program = EXEC_TEMPLATE_3X.render(
                indent="  ",
                is_profiler=True,
                is_bias=is_bias,
                is_bias_add=is_bias_add,
                instance_name=instance_name,
                dtype=dtype,
                # Profiler only measures latency; descale (alpha)/beta are irrelevant to
                # timing, so bake 1.0 regardless of fp8/f16.
                alpha_expr="1.0f",
                beta_expr="1.0f",
                # fp8 fusion decouples the per-channel bias to f16 (ElementBias); cast to
                # half so the arg type matches (bias_ptr is void*, so this is safe even
                # though the profiler's dummy bias tensor is E4M3 -- timing only).
                bias_elem=("cutlass::half_t" if _op_is_fp8 else f"{instance_name}_ElemC"),
            )
            instance = INSTANCE_TEMPLATE_3X.render(
                config_name=config_name,
                name=instance_name,
                config=config,
            )
            function = FUNCTION_TEMPLATE_3X.render(
                is_bias=is_bias,
                is_bias_add=is_bias_add,
                function_name=function_name,
                shape_function="",
                exec_paths=exec_program,
            )
            op_source = SRC_TEMPLATE_3X.render(
                extra_header="",
                instances=instance,
                functions=function,
            )
        else:
            config_name = extract_config_name(config)
            exec_program = EXEC_TEMPLATE.render(
                indent="  ",
                is_profiler=True,
                is_bias=is_bias,
                is_bias_add=is_bias_add,
                instance_name=instance_name,
                dtype=dtype,
            )
            instance = INSTANCE_TEMPLATE.render(
                config_name=config_name,
                name=instance_name,
                config=config,
            )
            function = FUNCTION_TEMPLATE.render(
                is_bias=is_bias,
                is_bias_add=is_bias_add,
                is_transpose=is_transpose,
                is_depthwise=is_depthwise,
                function_name=function_name,
                shape_function="",
                exec_paths=exec_program,
            )
            op_source = SRC_TEMPLATE.render(
                is_transpose=is_transpose,
                is_depthwise=is_depthwise,
                extra_header=extra_header,
                instances=instance,
                functions=function,
            )

        func_call = FUNC_CALL_TEMPLATE.render(
            indent="  ",
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            func_name=function_name,
            in_ptr="x.device_data()",
            weight_ptr="w.device_data()",
            out_ptr="y.device_data()",
            **func_call_extra_args,
            p_batch="&NI",
            p_out_ch="&CO",
            p_in_ch="&CI",
            p_kernel_h="&KH",
            p_kernel_w="&KW",
            p_in_h="&HI",
            p_in_w="&WI",
            p_out_batch="&NO",
            p_out_h="&HO",
            p_out_w="&WO",
            strideh="strideh",
            dilationh="dilationh",
            padh="padh",
            stridew="stridew",
            dilationw="dilationw",
            padw="padw",
        )
        benchmark_tmpl = BENCHMARK_TEMPLATE_3X if is_3x else BENCHMARK_TEMPLATE
        benchmark = benchmark_tmpl.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            instance_name_base=instance_name_base,
            function_name=function_name,
            func_call=func_call,
            instance_name=instance_name,
        )

        profiler_benchmarks[function_name] = PROFILER_BENCHMARK_TEMPLATE.render(
            op_source=op_source,
            benchmark=benchmark,
            instance_name=instance_name,
        )

        benchmark_instance = BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            conv_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            ni="NI",
            hi="HI",
            wi="WI",
            ci="CI",
            co="CO",
            kh="KH",
            kw="KW",
            no="NO",
            ho="HO",
            wo="WO",
            strideh="SH",
            dilationh="DH",
            padh="PH",
            stridew="SW",
            dilationw="DW",
            padw="PW",
        )
        benchmark_instances.append(benchmark_instance)

        benchmark_decl = BENCHMARK_DECL_TEMPLATE.render(
            function_name=function_name,
        )
        benchmark_decls.append(benchmark_decl)

    shape_func = shape_template.render(
        indent="  ",
        dtype="int64_t ",
        div="/",
        x_dim0="batch",
        x_dim1="in_h",
        x_dim2="in_w",
        x_dim3="in_ch",
        w_dim0="out_ch",
        w_dim1="kernel_h",
        w_dim2="kernel_w",
        strideh="strideh",
        dilateh="dilationh",
        padh="padh",
        stridew="stridew",
        dilatew="dilationw",
        padw="padw",
    )
    profiler_main_code = PROFILER_MAIN_TEMPLATE.render(
        shape_func=shape_func,
        benchmark_decls="\n".join(benchmark_decls),
        benchmark_instances="\n".join(benchmark_instances),
    )

    code = {profiler_filename: profiler_main_code}
    for benchmark_filename, benchmark_code in profiler_benchmarks.items():
        code[benchmark_filename] = benchmark_code

    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    add_profiler(file_pairs, workdir, op_type, profiler_filename, code)

    # build
    return build_profiler(file_pairs)


def extract_config_name(config):
    """Extracts config name from a given config."""
    pattern = re.compile(r"\s*using\s(.*?)\s=")
    decl = config.split("\n")[2]
    match = pattern.match(decl)
    if match is None:
        raise RuntimeError("Invalid config: \n" + config)
    return match.groups()[0]


def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
    f_emit_instance=emit_instance,
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
):
    """Function definition codegen."""
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    is_3x = any(
        getattr(op, "is_3x", False) for op in op_instance.values()
    )

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        op = op_instance[value]
        emitted_instance = f_emit_instance(op)
        if value not in inst_def_flag:
            inst_def_flag.add(value)
            config = emitted_instance
        else:
            config = ""
        if getattr(op, "is_3x", False):
            config_name = op.procedural_name() + "_base"
            inst = INSTANCE_TEMPLATE_3X.render(
                config=config,
                name=fname,
                config_name=config_name,
            )
        else:
            inst = INSTANCE_TEMPLATE.render(
                config=config,
                name=fname,
                config_name=extract_config_name(emitted_instance),
            )
        instances[key] = inst
        instance_decl += inst

    backend_spec = CUDASpec()
    in_dtype = func_attrs["inputs"][0]._attrs["dtype"]
    dtype = _conv_lib_dtype(backend_spec, in_dtype)
    # fp8 conv: the accumulator must be descaled by act_scale_inv * w_scale_inv
    # (per-tensor dequant) in the epilogue before bias/relu/residual. That scalar
    # is baked as an epilogue alpha literal from func_attrs["fp8_descale"], which
    # the driver sets from per-conv calibration (act amax) and the quantized
    # weight's |W|max. fp16 conv keeps alpha=1.0f.
    is_fp8_conv = in_dtype == "float8_e4m3"
    if is_fp8_conv:
        _descale = float(func_attrs.get("fp8_descale", 1.0))
        alpha_expr = f"{_descale!r}f"
        # beta scales the residual source. fp16 residual (partial fusion / conv5) -> 1.
        # full-fusion E4M3 residual -> out_scale/residual_scale (func_attrs["fp8_res_beta"]).
        _beta = float(func_attrs.get("fp8_res_beta", 1.0))
        beta_expr = f"{_beta!r}f"
        # fp8 bias is decoupled to f16 (ElementBias) in make_fusion_cpp.
        bias_elem = "cutlass::half_t"
    else:
        alpha_expr = "1.0f"
        beta_expr = "1.0f"
        bias_elem = None  # -> per-instance ElemC below
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        w_dim0="*out_ch",
        w_dim1="*kernel_h",
        w_dim2="*kernel_w",
        strideh="strideh",
        dilateh="dilationh",
        padh="padh",
        stridew="stridew",
        dilatew="dilationw",
        padw="padw",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
        y_dim3="*out_ch",
    )
    shape_func = shape_eval_func + shape_save_func

    exec_tmpl = EXEC_TEMPLATE_3X if is_3x else EXEC_TEMPLATE
    exec_paths = ""
    for key in instances:
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_tmpl.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            indent=" " * 4,
            instance_name=fname,
            dtype=dtype,
            alpha_expr=alpha_expr,
            beta_expr=beta_expr,
            bias_elem=(bias_elem if bias_elem is not None else f"{fname}_ElemC"),
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst

    if is_3x:
        function = FUNCTION_TEMPLATE_3X.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            function_name=func_name,
            shape_function=shape_func,
            exec_paths=exec_paths,
        )
        return SRC_TEMPLATE_3X.render(
            extra_header="",
            instances=instance_decl,
            functions=function,
        )

    function = FUNCTION_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        is_transpose=is_transpose,
        is_depthwise=is_depthwise,
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
    )

    return SRC_TEMPLATE.render(
        is_transpose=is_transpose,
        is_depthwise=is_depthwise,
        extra_header=extra_header,
        instances=instance_decl,
        functions=function,
    )


def gen_function_decl(
    func_attrs,
    is_bias=False,
    is_bias_add=False,
):
    func_name = func_attrs["name"]

    return FUNC_DECL_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        func_name=func_name,
    )


def gen_function_call(
    func_attrs,
    indent="  ",
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]

    func_call_extra_args = {}
    if is_bias:
        b = func_attrs["inputs"][2]
        func_call_extra_args = {
            "bias_ptr": b._attrs["name"],
        }
    elif is_bias_add:
        b = func_attrs["inputs"][2]
        r = func_attrs["inputs"][3]
        func_call_extra_args = {
            "bias_ptr": b._attrs["name"],
            "res_ptr": r._attrs["name"],
        }

    out_ch = wshape[-1]._attrs["name"] if is_transpose else wshape[0]._attrs["name"]
    return FUNC_CALL_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        out_ptr=y._attrs["name"],
        **func_call_extra_args,
        p_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + out_ch,
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_kernel_h="&" + wshape[1]._attrs["name"],
        p_kernel_w="&" + wshape[2]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        strideh=(
            func_attrs["stride"]
            if isinstance(func_attrs["stride"], int)
            else func_attrs["stride"][0]
        ),
        dilationh=(
            func_attrs["dilate"]
            if isinstance(func_attrs["dilate"], int)
            else func_attrs["dilate"][0]
        ),
        padh=(
            func_attrs["pad"]
            if isinstance(func_attrs["pad"], int)
            else func_attrs["pad"][0]
        ),
        stridew=(
            func_attrs["stride"]
            if isinstance(func_attrs["stride"], int)
            else func_attrs["stride"][1]
        ),
        dilationw=(
            func_attrs["dilate"]
            if isinstance(func_attrs["dilate"], int)
            else func_attrs["dilate"][1]
        ),
        padw=(
            func_attrs["pad"]
            if isinstance(func_attrs["pad"], int)
            else func_attrs["pad"][1]
        ),
        indent=indent,
    )


def _cal_align_ab(x_shape: List[int], dtype="float16") -> int:
    """Returns input alignment."""
    k = x_shape[3]  # CI
    return alignment.find_max_alignment(k, dtype)


def function_filter(
    cfg,
    func_attrs,
    x_shape,
):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    dtype = func_attrs["inputs"][0]._attrs["dtype"]
    ab_alignment = _cal_align_ab(x_shape, dtype=dtype)

    tmp = cfg.split("_")
    align_c = int(tmp[-1])
    align_ab = int(tmp[-2])

    if align_c != func_attrs["epilogue_alignment"]:
        return False
    if align_ab != ab_alignment:
        return False

    return True
