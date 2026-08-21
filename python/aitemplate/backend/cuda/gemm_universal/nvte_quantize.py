#  Backend codegen for nvte_quantize: wraps TransformerEngine's nvte_quantize
#  (Y = quantize(X) to E4M3, Y = X*scale). fp16 in, FP8 out (+ amax/scale_inv scratch).
#  Link/headers same as the other nvte ops. See docs/te_fp8_impl_plan.md.
import importlib.util
import os

import jinja2

from aitemplate.backend import registry


def _nvte_include_dir():
    env = os.environ.get("NVTE_INCLUDE")
    if env:
        return env
    spec = importlib.util.find_spec("transformer_engine")
    if spec and spec.submodule_search_locations:
        return os.path.join(spec.submodule_search_locations[0], "common", "include")
    return ""


_NVTE_INCLUDE = _nvte_include_dir()

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <vector>
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/cast.h"

using transformer_engine::TensorWrapper;
using transformer_engine::DType;

// Y[rows,C] (E4M3) = quantize(X[rows,C] fp16) = X*scale. amax + scale_inv (4B each) at
// the front of workspace are written by TE (scale_inv must be allocated); not read back.
void {{func_name}}(void* x_ptr, void* y_ptr, int64_t rows, int64_t C,
                   void* scale_ptr, uint8_t* workspace, cudaStream_t stream) {
  float* amax = reinterpret_cast<float*>(workspace);
  float* scale_inv = reinterpret_cast<float*>(workspace + 4);
  TensorWrapper X(x_ptr, std::vector<size_t>{(size_t)rows, (size_t)C}, DType::kFloat16);
  TensorWrapper Y(y_ptr, std::vector<size_t>{(size_t)rows, (size_t)C}, DType::kFloat8E4M3,
                  /*amax=*/amax, /*scale=*/reinterpret_cast<float*>(scale_ptr),
                  /*scale_inv=*/scale_inv);
  nvte_quantize(X.data(), Y.data(), stream);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(void*, void*, int64_t, int64_t, void*, uint8_t*, cudaStream_t);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{y_ptr}}, {{rows_expr}}, {{c}}, {{scale}},
{{indent}}    global_workspace_, stream
{{indent}});
"""
)


@registry.reg("cuda.nvte_quantize.gen_function")
def nvte_quantize_gen_function(func_attrs):
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"], nvte_include=_NVTE_INCLUDE)


@registry.reg("cuda.nvte_quantize.func_decl")
def nvte_quantize_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.nvte_quantize.func_call")
def nvte_quantize_gen_function_call(func_attrs, indent="  "):
    x, scale = func_attrs["inputs"][0], func_attrs["inputs"][1]
    y = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    c = x_shape[-1]._attrs["values"][0]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        y_ptr=y._attrs["name"],
        rows_expr=rows_expr,
        c=c,
        scale=scale._attrs["name"],
    )
