#  Backend codegen for nvte_rmsnorm: wraps TransformerEngine's nvte_rmsnorm_fwd
#  (Z = RMSNorm(X)*gamma) with its two-pass workspace-size query. fp16 today.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <vector>
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/normalization.h"

using transformer_engine::TensorWrapper;
using transformer_engine::DType;
// NVTEShape is a global C type (not in the transformer_engine namespace).

// Z[rows,C] = RMSNorm(X[rows,C]) * gamma[C]. workspace holds rsigma [rows] fp32 then
// the (tiny) nvte scratch; the required scratch size is queried (pass 1, empty ws).
void {{func_name}}(void* x_ptr, void* gamma_ptr, void* z_ptr,
                   int64_t rows, int64_t C, float eps,
                   uint8_t* workspace, cudaStream_t stream) {
  int dev = 0, sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
  TensorWrapper X(x_ptr, std::vector<size_t>{(size_t)rows, (size_t)C}, DType::kFloat16);
  TensorWrapper G(gamma_ptr, std::vector<size_t>{(size_t)C}, DType::kFloat16);
  TensorWrapper Z(z_ptr, std::vector<size_t>{(size_t)rows, (size_t)C}, DType::kFloat16);
  float* rsigma = reinterpret_cast<float*>(workspace);
  TensorWrapper R(rsigma, std::vector<size_t>{(size_t)rows}, DType::kFloat32);
  uint8_t* ws2 = workspace + (size_t)rows * sizeof(float);
  // pass 1: query nvte scratch size (empty workspace tensor)
  TensorWrapper wsq;
  nvte_rmsnorm_fwd(X.data(), G.data(), eps, Z.data(), R.data(), wsq.data(), sm,
                   /*zero_centered_gamma=*/false, stream);
  NVTEShape wshape = wsq.shape();
  std::vector<size_t> wsv;
  for (size_t i = 0; i < wshape.ndim; ++i) wsv.push_back(wshape.data[i]);
  TensorWrapper W(ws2, wsv, wsq.dtype());
  // pass 2: compute
  nvte_rmsnorm_fwd(X.data(), G.data(), eps, Z.data(), R.data(), W.data(), sm,
                   /*zero_centered_gamma=*/false, stream);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(void*, void*, void*, int64_t, int64_t, float, uint8_t*, cudaStream_t);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{x_ptr}}, {{gamma_ptr}}, {{z_ptr}},
{{indent}}    {{rows_expr}}, {{c}}, {{eps}}f,
{{indent}}    global_workspace_, stream
{{indent}});
"""
)


@registry.reg("cuda.nvte_rmsnorm.gen_function")
def nvte_rmsnorm_gen_function(func_attrs):
    return FUNC_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.nvte_rmsnorm.func_decl")
def nvte_rmsnorm_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.nvte_rmsnorm.func_call")
def nvte_rmsnorm_gen_function_call(func_attrs, indent="  "):
    x, gamma = func_attrs["inputs"]
    z = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in x_shape[:-1]) or "1"
    c = x_shape[-1]._attrs["values"][0]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        x_ptr=x._attrs["name"],
        gamma_ptr=gamma._attrs["name"],
        z_ptr=z._attrs["name"],
        rows_expr=rows_expr,
        c=c,
        eps=func_attrs["eps"],
    )
