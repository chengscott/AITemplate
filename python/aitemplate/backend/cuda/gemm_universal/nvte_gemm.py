#  Backend codegen for the nvte_gemm op: emits a C++ wrapper that calls
#  TransformerEngine's nvte_cublas_gemm (Y = X @ W^T). fp16 today.
#
#  Link needs: -ltransformer_engine -lcublasLt -lcublas -lnvrtc -lcuda -lcudart
#  Headers: NVTE_INCLUDE (default = the te_build source include dir).
import importlib.util
import os

import jinja2

from aitemplate.backend import registry


def _nvte_include_dir():
    """TE C++ header dir. NVTE_INCLUDE overrides; else derive from the installed
    transformer_engine package (find_spec avoids importing TE just to locate it)."""
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
// TE headers resolved via -I{{nvte_include}} (added to NVCC_APPEND_FLAGS); their
// internal includes are relative so the -I is required, not just an absolute path.
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/gemm.h"

using transformer_engine::TensorWrapper;
using transformer_engine::MatmulConfigWrapper;
using transformer_engine::DType;

// Y[M,N] = X[M,K] @ W[N,K]^T  (nn.Linear). A=W (transa), B=X. Uses nvte_cublas_gemm_v2
// (v1 is deprecated): D = alpha*op(A)*op(B) + beta*C; alpha=1, beta=0 so C is unused
// (pass D). Default MatmulConfigWrapper = no bias/gelu, auto SM count.
void {{func_name}}(void* x_ptr, void* w_ptr, void* y_ptr,
                   int64_t M, int64_t K, int64_t N,
                   uint8_t* workspace, int64_t ws_bytes, cudaStream_t stream) {
  TensorWrapper A(w_ptr, std::vector<size_t>{(size_t)N, (size_t)K}, DType::kFloat16);
  TensorWrapper B(x_ptr, std::vector<size_t>{(size_t)M, (size_t)K}, DType::kFloat16);
  TensorWrapper D(y_ptr, std::vector<size_t>{(size_t)M, (size_t)N}, DType::kFloat16);
  TensorWrapper ws(workspace, std::vector<size_t>{(size_t)ws_bytes}, DType::kByte);
  MatmulConfigWrapper config;
  const float alpha = 1.0f, beta = 0.0f;
  nvte_cublas_gemm_v2(/*transa=*/1, /*transb=*/0, &alpha, A.data(), B.data(),
                      &beta, /*C=*/D.data(), /*D=*/D.data(), ws.data(),
                      config, stream);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(void*, void*, void*, int64_t, int64_t, int64_t,
                   uint8_t*, int64_t, cudaStream_t);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{w_ptr}}, {{y_ptr}},
{{indent}}    {{m_expr}}, {{k}}, {{n}},
{{indent}}    global_workspace_, {{ws_bytes}}L, stream
{{indent}});
"""
)


@registry.reg("cuda.nvte_gemm.gen_function")
def nvte_gemm_gen_function(func_attrs):
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"], nvte_include=_NVTE_INCLUDE
    )


@registry.reg("cuda.nvte_gemm.func_decl")
def nvte_gemm_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.nvte_gemm.func_call")
def nvte_gemm_gen_function_call(func_attrs, indent="  "):
    x, w = func_attrs["inputs"]
    y = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]
    # M = product of X's leading dims (batch/seq); K, N static.
    m_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    k = x_shape[-1]._attrs["values"][0]
    n = w._attrs["shape"][0]._attrs["values"][0]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        w_ptr=w._attrs["name"],
        y_ptr=y._attrs["name"],
        m_expr=m_expr,
        k=k,
        n=n,
        ws_bytes=func_attrs["workspace"],
    )
