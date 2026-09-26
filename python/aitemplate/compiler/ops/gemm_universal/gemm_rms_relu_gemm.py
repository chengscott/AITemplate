"""FP16 residual projection followed by RMSNorm/ReLU and an outer projection."""

import math
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

from .native_fusion import NativeFusionProfiler


class gemm_rms_relu_gemm(NativeFusionProfiler, Operator):
    """Fuse two bias-free projections across RMSNorm/ReLU.

    The positive RMS scale moves to the second GEMM epilogue. The first
    GEMM rounds its residual sum to FP16, then stores ReLU(sum * gamma).
    This changes intermediate rounding relative to a separate RMSNorm.
    """

    def __init__(self, eps=1e-6, profile_rows=None):
        """Optionally tune native kernels at exact flattened ``profile_rows``."""
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMS epsilon must be positive and finite")
        self._attrs.update(
            op="gemm_rms_relu_gemm",
            has_profiler=True,
            eps=float(eps),
            profile_rows=None if profile_rows is None else tuple(profile_rows),
        )

    def __call__(self, x, weight, residual, gamma, up_weight, outer):
        inputs = [x, weight, residual, gamma, up_weight, outer]
        if any(t.dtype() != "float16" for t in inputs):
            raise ValueError("Fused RMS projections require FP16 inputs")
        if len(x.shape()) < 2 or x.shape()[-1]._attrs["values"] != [576]:
            raise ValueError("Input must have static width 576")
        for t, expected in (
            (weight, [[192], [576]]),
            (gamma, [[192]]),
            (up_weight, [[384], [192]]),
        ):
            if [d._attrs["values"] for d in t.shape()] != expected:
                raise ValueError(f"Invalid parameter shape; expected {expected}")
        for t, width in ((residual, 192), (outer, 384)):
            if t.shape()[:-1] != x.shape()[:-1] or t.shape()[-1]._attrs["values"] != [
                width
            ]:
                raise ValueError("Residual row dimensions and static width must match")
        max_rows = math.prod(d.upper_bound() for d in x.shape()[:-1])
        if any(d.lower_bound() < 1 for d in x.shape()[:-1]) or max_rows > 2**31 - 1:
            raise ValueError("Row count must be positive and fit int32")
        self._attrs.update(
            inputs=inputs,
            workspace=max_rows * 384 + ((max_rows + 127) // 128) * 128 * 4 + 255,
        )
        self._set_depth()
        out = Tensor(outer.shape(), src_ops={self}, dtype="float16")
        self._attrs["outputs"] = [out]
        return out

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"], "profile_rows": self._attrs["profile_rows"]}

    def gen_function(self):
        target = backend.target.Target.current()
        return registry.get(f"{target.name()}.gemm_rms_relu_gemm.gen_function")(
            self._attrs
        )
