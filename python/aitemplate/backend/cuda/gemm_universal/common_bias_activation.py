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
Common codegen functions for gemm_bias_activation.
"""

import jinja2
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common, common_bias, gemm_rcr
from aitemplate.backend.cuda.gemm_universal.layout import RCR
from aitemplate.backend.target import Target

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


class _LinCombPerColBiasEltActFunctor:
    """SM100 EVT epilogue: D = act(alpha*acc + beta*C + per-column bias).

    The activation counterpart of gemm_rcr_bias._LinCombPerColBiasFunctor -- emitted as
    the collective epilogue's fusion Operation for the CUTLASS 3.x SM100 path (SM100 has
    no BiasElementwise schedule). ``activation`` is the cutlass epilogue activation tag,
    e.g. ``cutlass::epilogue::thread::ReLu``.
    """

    value = 10101  # profile-cache key sentinel (distinct from the plain-bias functor)

    def __init__(
        self, activation, element_output, element_compute, element_bias, element_source, element_scalar
    ):
        self.activation = activation
        self.element_output = element_output
        self.element_compute = element_compute
        self.element_bias = element_bias
        self.element_source = element_source
        self.element_scalar = element_scalar

    def emit_declaration(self):
        return (
            "cutlass::epilogue::fusion::LinCombPerColBiasEltAct<"
            f"{self.activation}, {self.element_output}, {self.element_compute}, "
            f"{self.element_bias}, {self.element_source}, {self.element_scalar}>"
        )


EXTRA_CODE_HEADER = jinja2.Template(
    """
using elem_input_type = {{elem_input_type}};
using elem_output_type = {{elem_output_type}};
"""
)


# Shared SM100 (Blackwell) 3.x problem args for all bias+activation gemms: the
# LinCombPerColBiasEltAct fusion Arguments (activation is compile-time in the functor,
# so the runtime args are identical to plain LinCombPerColBias). Non-transposed problem;
# per-column bias broadcast. Used for arch=="100" in place of each op's SM90 template.
SM100_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    {  // MainloopArguments mainloop (non-transposed; bias+act fused via EVT)
    ({{elem_input_type}}*)(a_ptr),                               // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideA dA
    ({{elem_input_type}}*)(b_ptr),                               // ElementB const* ptr_B
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideB dB
    },
    {  // EpilogueArguments (LinCombPerColBiasEltAct<Act>: act(alpha*acc + beta*C + bias))
        {                                                        // thread (fusion args)
            ElementComputeEpilogue(1),                           // alpha
            ElementComputeEpilogue(0),                           // beta (no residual C)
            nullptr,                                             // alpha_ptr
            nullptr,                                             // beta_ptr
            {cute::Int<0>{}, cute::Int<0>{}, int64_t(0)},        // StrideAlpha dAlpha
            {cute::Int<0>{}, cute::Int<0>{}, int64_t(0)},        // StrideBeta dBeta
            ({{elem_input_type}}*)(bias_ptr),                    // ElementBias const* bias_ptr
            {cute::Int<0>{}, cute::Int<1>{}, int64_t(0)},        // StrideBias dBias (per-col)
        },
        nullptr,                                                 // ElementC const* ptr_C
        {cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}},        // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
"""
)


def gemm_rcr_config(
    func_attrs,
    dtype="float16",
    include_cutlass_3x_ops=False,
    activation_tag=None,
):
    common.make_fproc(
        func_attrs=func_attrs,
        layout=RCR,
        include_cutlass_3x_ops=include_cutlass_3x_ops,
    )

    import cutlass_lib

    lib = cutlass_lib.library
    bias_map = lib.EpilogueScheduleBiasElementwiseMapping
    evt = Target.current()._arch == "100"  # activation EVT: SM100 only (Sm90 relu FusionCallbacks::Arguments is nested/tuple, not the flat LinCombPerColBias form -- genuine cutlass mismatch, see gb200-sm100-port memory)
    drop = []
    for name, op in func_attrs["op_instance"].items():
        if common.has_tma_epilogue(op):
            if evt:
                # SM90a + SM100: fuse bias+activation via the EVT epilogue
                # (LinCombPerColBiasEltAct), non-transposed. Ops without a mapped activation
                # (e.g. mul) drop -> SM80 fallback.
                if activation_tag is not None:
                    op.epilogue_functor = _LinCombPerColBiasEltActFunctor(
                        activation=activation_tag,
                        element_output=lib.DataTypeTag[op.D.element],
                        element_compute=lib.DataTypeTag[op.element_epilogue],
                        element_bias=lib.DataTypeTag[op.A.element],
                        element_source=lib.DataTypeTag[op.A.element],
                        element_scalar=lib.DataTypeTag[op.element_epilogue],
                    )
                else:
                    drop.append(name)
                continue
            # legacy SM90 bias-via-schedule (transposed problem):
            op.C.element = lib.DataType.void
            op.C.layout = lib.LayoutType.ColumnMajor
            op.D.layout = lib.LayoutType.ColumnMajor
            op.epilogue_schedule = bias_map[op.epilogue_schedule]
    for name in drop:
        del func_attrs["op_instance"][name]


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    problem_args_template,
    problem_args_template_cutlass_3x=None,
    extra_code="",
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    extra_code_header = EXTRA_CODE_HEADER.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    tmpl_3x = (
        SM100_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X
        if Target.current()._arch == "100"
        else problem_args_template_cutlass_3x
    )
    return gemm_rcr.common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_bias.SRC_TEMPLATE,
        problem_args_template=problem_args_template,
        problem_args_template_cutlass_3x=tmpl_3x,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
        extra_code="\n\n".join([extra_code_header, extra_code]),
    )


def gen_function(
    func_attrs,
    problem_args_template,
    exec_cond_template,
    dim_info_dict,
    problem_args_template_cutlass_3x=None,
    extra_code="",
):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    problem_args = problem_args_template.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    problem_args_cutlass_3x = ""
    if problem_args_template_cutlass_3x is not None:
        tmpl_3x = (
            SM100_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X
            if Target.current()._arch == "100"
            else problem_args_template_cutlass_3x
        )
        problem_args_cutlass_3x = tmpl_3x.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        )
    extra_code_header = EXTRA_CODE_HEADER.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common.gen_function(
        func_attrs=func_attrs,
        src_template=common_bias.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N",
            output_accessor=func_attrs["output_accessors"][0],
        ),
        extra_code="\n\n".join([extra_code_header, extra_code]),
    )


def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        bias_ptr_arg=bias._attrs["name"],
    )
