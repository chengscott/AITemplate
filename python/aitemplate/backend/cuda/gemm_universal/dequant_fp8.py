#  Backend codegen for `dequant_fp8`: fold the fp8 gemm's fp32 accumulator back to f16.
#    y[r,n] = act( acc[r,n] * (scale_x[r] * scale_w[0]) + bias[n] + residual[r,n] )
#  One warp per row; scale_x is per-token (from quantize_to_fp8 / im2col_fp8), scale_w
#  per-tensor (folded at export); bias, residual and ReLU are optional (they recover the
#  f16 gemm/conv epilogue fusions). See compiler/ops/gemm_universal/dequant_fp8.py.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
__global__ void {{func_name}}_kernel(const float* __restrict__ acc,
                                     const float* __restrict__ scale_x,
                                     const float* __restrict__ scale_w,
                                     const __half* __restrict__ bias,
                                     const __half* __restrict__ residual,
                                     __half* __restrict__ y,
                                     long long rows, int N) {
  const int warps_per_cta = blockDim.x >> 5;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
{% if scalar_scale %}
  const float s = scale_x[0] * scale_w[0];
{% else %}
  const float s = scale_x[row] * scale_w[0];
{% endif %}
  const long long base = row * (long long)N;
  for (int n = lane; n < N; n += 32) {
    float v = acc[base + n] * s;
{% if has_bias %}
    v += __half2float(bias[n]);
{% endif %}
{% if has_residual %}
    v += __half2float(residual[base + n]);
{% endif %}
{% if relu %}
    v = v > 0.f ? v : 0.f;
{% endif %}
    y[base + n] = __float2half(v);
  }
}
}  // namespace

void {{func_name}}(const void* acc_ptr, const void* scale_x_ptr, const void* scale_w_ptr,
                   const void* bias_ptr, const void* residual_ptr, void* y_ptr,
                   int64_t rows, int64_t N, cudaStream_t stream) {
  constexpr int BLK = 128;  // 4 warps/CTA, one row per warp
  const unsigned int grid = (unsigned int)((rows + (BLK >> 5) - 1) / (BLK >> 5));
  {{func_name}}_kernel<<<grid, BLK, 0, stream>>>(
      reinterpret_cast<const float*>(acc_ptr),
      reinterpret_cast<const float*>(scale_x_ptr),
      reinterpret_cast<const float*>(scale_w_ptr),
      reinterpret_cast<const __half*>(bias_ptr),
      reinterpret_cast<const __half*>(residual_ptr),
      reinterpret_cast<__half*>(y_ptr), (long long)rows, (int)N);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, const void*, const void*, const void*, const void*, "
    "void*, int64_t, int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{acc_ptr}}, {{scale_x_ptr}}, {{scale_w_ptr}}, {{bias_ptr}}, {{residual_ptr}},
{{indent}}    {{y_ptr}}, {{rows_expr}}, {{n}}, stream
{{indent}});
"""
)


@registry.reg("cuda.dequant_fp8.gen_function")
def gen_function(func_attrs):
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        has_bias=func_attrs.get("has_bias", False),
        has_residual=func_attrs.get("has_residual", False),
        relu=func_attrs.get("relu", False),
        scalar_scale=func_attrs.get("scalar_scale", False),
    )


@registry.reg("cuda.dequant_fp8.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.dequant_fp8.func_call")
def gen_function_call(func_attrs, indent="  "):
    acc = func_attrs["inputs"][0]
    scale_x = func_attrs["inputs"][1]
    scale_w = func_attrs["inputs"][2]
    idx = 3
    if func_attrs.get("has_bias", False):
        bias_ptr = func_attrs["inputs"][idx]._attrs["name"]
        idx += 1
    else:
        bias_ptr = "nullptr"
    if func_attrs.get("has_residual", False):
        residual_ptr = func_attrs["inputs"][idx]._attrs["name"]
        idx += 1
    else:
        residual_ptr = "nullptr"
    y = func_attrs["outputs"][0]
    acc_shape = acc._attrs["shape"]
    rows_expr = " * ".join(d._attrs["name"] for d in acc_shape[:-1]) or "1"
    n = acc_shape[-1]._attrs["values"][0]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        acc_ptr=acc._attrs["name"],
        scale_x_ptr=scale_x._attrs["name"],
        scale_w_ptr=scale_w._attrs["name"],
        bias_ptr=bias_ptr,
        residual_ptr=residual_ptr,
        y_ptr=y._attrs["name"],
        rows_expr=rows_expr,
        n=n,
    )
