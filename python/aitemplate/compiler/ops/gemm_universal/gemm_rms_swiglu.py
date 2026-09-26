"""FP16 projection, RMS scaling and SwiGLU with an optional residual producer."""

import math
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor, IntImm

from .native_fusion import NativeFusionProfiler


class gemm_rms_swiglu(NativeFusionProfiler, Operator):
    """SwiGLU((X @ W.T) * RMS(X)); W contains the folded normalization gain.

    With residual/projection_weight, first compute
    X = half(input @ projection_weight.T + residual), and return (X, output).
    Projection biases are not supported. The fused FFN consumes FP32 GEMM
    accumulators directly, so rounding can differ from a separate FP16 GEMM.
    """

    def __init__(self, eps=1e-6, profile_rows=None):
        """Optionally tune native kernels at exact flattened ``profile_rows``."""
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMS epsilon must be positive and finite")
        self._attrs.update(
            op="gemm_rms_swiglu",
            has_profiler=True,
            eps=float(eps),
            profile_rows=None if profile_rows is None else tuple(profile_rows),
        )

    def __call__(self, x, weight, residual=None, projection_weight=None):
        joint = residual is not None
        if joint != (projection_weight is not None):
            raise ValueError("Residual and projection weight must be supplied together")
        inputs = [x, weight] + ([residual, projection_weight] if joint else [])
        if any(t.dtype() != "float16" for t in inputs):
            raise ValueError("Fused RMS GEMM requires FP16 inputs")
        if len(x.shape()) < 2 or x.shape()[-1]._attrs["values"] != [192]:
            raise ValueError("Fused RMS GEMM requires static input width 192")
        if [d._attrs["values"] for d in weight.shape()] != [[1152], [192]]:
            raise ValueError("SwiGLU weight must have shape [1152, 192], gate then up")
        if joint:
            if residual.shape() != x.shape():
                raise ValueError("Residual must match the input shape")
            if [d._attrs["values"] for d in projection_weight.shape()] != [
                [192],
                [192],
            ]:
                raise ValueError(
                    "Residual projection weight must have shape [192, 192]"
                )
        max_rows = math.prod(d.upper_bound() for d in x.shape()[:-1])
        if any(d.lower_bound() < 1 for d in x.shape()[:-1]):
            raise ValueError("Fused RMS GEMM requires nonempty row dimensions")
        if max_rows > 2**31 - 1:
            raise ValueError("CuTe GEMM row count exceeds int32")
        # Native reductions store a complete final row tile.
        self._attrs.update(
            inputs=inputs, joint=joint, workspace=((max_rows + 127) // 128) * 128 * 4
        )
        self._set_depth()
        out = Tensor(
            list(x.shape()[:-1]) + [IntImm(576)], src_ops={self}, dtype="float16"
        )
        if joint:
            updated = Tensor(x.shape(), src_ops={self}, dtype="float16")
            self._attrs["outputs"] = [updated, out]
            return updated, out
        self._attrs["outputs"] = [out]
        return out

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"], "profile_rows": self._attrs["profile_rows"]}

    def gen_function(self):
        target = backend.target.Target.current()
        return registry.get(f"{target.name()}.gemm_rms_swiglu.gen_function")(
            self._attrs
        )
