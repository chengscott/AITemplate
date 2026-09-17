#  Fused dequantization epilogue for the fp8 gemm path.
#
#  Given the raw fp32 accumulator from gemm_rcr_fp8, the per-token activation scale
#  (scale_x [.., 1] f32) and the per-tensor weight scale (scale_w [1] f32), produce f16:
#
#     y[r,n] = act( acc[r,n] * scale_x[r] * scale_w[0] + bias[n] + residual[r,n] )
#
#  bias ([N] f16), residual (same shape as acc, f16) and act (ReLU) are all optional --
#  they let a single dequant kernel recover the f16 gemm/conv epilogue fusions
#  (gemm_rcr_bias_add / Conv2dBiasAddRelu, ...). One warp per row streams acc once.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class dequant_fp8(Operator):
    """act(acc_f32[..,N] * scale_x[..,1] * scale_w[1] + bias[N] + residual[..,N]) -> f16."""

    def __init__(self, relu: bool = False, scalar_scale: bool = False) -> None:
        super().__init__()
        self._attrs["op"] = "dequant_fp8"
        self._attrs["has_profiler"] = False
        self._attrs["relu"] = bool(relu)
        # scalar_scale: scale_x is a single [1] scalar (per-tensor, conv path) rather than
        # a per-row [rows,1] vector (per-token, linear path).
        self._attrs["scalar_scale"] = bool(scalar_scale)

    def __call__(
        self,
        acc: Tensor,
        scale_x: Tensor,
        scale_w: Tensor,
        bias: Tensor = None,
        residual: Tensor = None,
    ):
        inputs = [acc, scale_x, scale_w]
        if bias is not None:
            inputs.append(bias)
        if residual is not None:
            inputs.append(residual)
        self._attrs["inputs"] = inputs
        self._attrs["has_bias"] = bias is not None
        self._attrs["has_residual"] = residual is not None
        self._set_depth()
        y = Tensor(list(acc._attrs["shape"]), src_ops={self}, dtype="float16")
        self._attrs["outputs"] = [y]
        return y

    def _get_op_attributes(self):
        return {"relu": self._attrs["relu"], "scalar_scale": self._attrs["scalar_scale"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
