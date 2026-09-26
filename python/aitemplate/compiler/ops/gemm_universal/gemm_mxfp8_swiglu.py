"""Block-scaled FFN fusion with constant-foldable gate/up weight packing."""

import math
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

from .native_fusion import NativeFusionProfiler


class _pack_mxfp8_swiglu(Operator):
    def __init__(self):
        super().__init__()
        self._attrs.update(op="pack_mxfp8_swiglu", has_profiler=False)

    def __call__(self, weight, scale):
        self._attrs["inputs"] = [weight, scale]
        self._set_depth()
        outputs = [
            Tensor(t.shape(), dtype=t.dtype(), src_ops={self}) for t in (weight, scale)
        ]
        self._attrs["outputs"] = outputs
        return outputs

    def gen_function(self):
        target = backend.target.Target.current()
        return registry.get(f"{target.name()}.pack_mxfp8_swiglu.gen_function")(
            self._attrs
        )


class gemm_mxfp8_swiglu(NativeFusionProfiler, Operator):
    """RMS-scaled SwiGLU FFN with MXFP8 weights and an FP16 residual.

    Input weights retain the standard gate-then-up format. A separate packing
    op interleaves gate/up rows and their block scales; constant folding moves
    that work out of inference when the weights are model constants.
    """

    def __init__(self, eps=1e-6, profile_rows=None):
        """Optionally tune native kernels at exact flattened ``profile_rows``."""
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMS epsilon must be positive and finite")
        self._attrs.update(
            op="gemm_mxfp8_swiglu",
            has_profiler=True,
            eps=float(eps),
            profile_rows=None if profile_rows is None else tuple(profile_rows),
        )

    def __call__(self, x, weight, scale, down_weight, down_scale):
        if (
            x.dtype() != "float16"
            or len(x.shape()) < 2
            or x.shape()[-1]._attrs["values"] != [192]
        ):
            raise ValueError("MXFP8 FFN requires an FP16 input of static width 192")
        for t, shape in (
            (weight, [[1152], [192]]),
            (scale, [[9216]]),
            (down_weight, [[192], [576]]),
            (down_scale, [[5120]]),
        ):
            if (
                t.dtype() != "float8_e4m3"
                or [d._attrs["values"] for d in t.shape()] != shape
            ):
                raise ValueError(f"Expected FP8 carrier tensor with shape {shape}")
        rows = math.prod(d.upper_bound() for d in x.shape()[:-1])
        if any(d.lower_bound() < 1 for d in x.shape()[:-1]) or rows > 2**31 - 1:
            raise ValueError("Row count must be positive and fit int32")
        weight, scale = _pack_mxfp8_swiglu()(weight, scale)
        self._attrs.update(
            inputs=[x, weight, scale, down_weight, down_scale],
            workspace=rows * 770 + ((rows + 127) // 128) * 3584 + 5 * 255,
        )
        self._set_depth()
        out = Tensor(x.shape(), dtype="float16", src_ops={self})
        self._attrs["outputs"] = [out]
        return out

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"], "profile_rows": self._attrs["profile_rows"]}

    def gen_function(self):
        target = backend.target.Target.current()
        return registry.get(f"{target.name()}.gemm_mxfp8_swiglu.gen_function")(
            self._attrs
        )
