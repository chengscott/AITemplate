"""Fuse a folded RMSNorm reduction with its FP16 activation consumer."""
import math
from aitemplate.compiler.ops.gemm_universal.swiglu import swiglu
from aitemplate.compiler.ops.gemm_universal.unpack_rope import unpack_rope


def _validate(source, projected, eps):
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("RMS epsilon must be finite and nonnegative")
    if source.dtype() != "float16" or projected.dtype() != "float16":
        raise ValueError("Fused RMS consumers require float16 inputs")
    if source.shape()[:-1] != projected.shape()[:-1]:
        raise ValueError("RMS source and projected tensor must have matching rows")
    channels = source.shape()[-1]
    if len(channels._attrs["values"]) != 1 or channels.value() % 8:
        raise ValueError("RMS source width must be static and divisible by eight")


class rms_swiglu(swiglu):
    """SwiGLU(projected * half(rsqrt(mean(source**2) + eps)))."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self._attrs.update(op="rms_swiglu", eps=float(eps))

    def __call__(self, projected, source):
        _validate(source, projected, self._attrs["eps"])
        if projected.shape()[-1].value() % 16:
            raise ValueError("Fused RMS SwiGLU requires an FFN width divisible by eight")
        return super().__call__(projected, source)

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"]}


class rms_unpack_rope(unpack_rope):
    """RMS statistics from source; scale, rotate and unpack projected QKV."""

    def __init__(self, heads, eps=1e-6):
        super().__init__(heads)
        self._attrs.update(op="rms_unpack_rope", eps=float(eps))

    def __call__(self, projected, cosb, sinb, source):
        _validate(source, projected, self._attrs["eps"])
        if len(projected.shape()) != 3 or len(projected.shape()[1]._attrs["values"]) != 1:
            raise ValueError("Fused RMS RoPE requires [batch, static sequence, packed channels]")
        if projected.shape()[-1].value() != 3 * self._attrs["heads"] * 16:
            raise ValueError("Fused RMS RoPE currently requires head dimension 16")
        expected = [1, projected.shape()[1].value(), self._attrs["heads"], 8]
        for trig in (cosb, sinb):
            if trig.dtype() != "float16" or [d._attrs["values"] for d in trig.shape()] != [[d] for d in expected]:
                raise ValueError("Fused RMS RoPE requires FP16 cosine/sine tables [1, sequence, heads, 8]")
        return super().__call__(projected, cosb, sinb, source)

    def _get_op_attributes(self):
        return {"heads": self._attrs["heads"], "eps": self._attrs["eps"]}
