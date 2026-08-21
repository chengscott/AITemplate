#  SwiGLU activation as an AITemplate op (ports TE's swiglu.cu / tex.swiglu into AIT).
#
#  Z = silu(X[..., :ffn]) * X[..., ffn:] over the packed fc1 output X: [..., 2*ffn] -> Z:
#  [..., ffn]. One inline elementwise kernel; consuming the packed input directly removes
#  the split op the generic (elementwise) SwiGLU needs. fp16.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class swiglu(Operator):
    """Z = silu(X[...,:ffn]) * X[...,ffn:], X: [..., 2*ffn] -> Z: [..., ffn]."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "swiglu"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor, rrms: Tensor = None) -> Tensor:
        """x: [..., 2*ffn] (fc1 output, gate first). Optional rrms [..,1]
        (rmsnorm-prologue): each row's gate/up is scaled by rrms before silu*mul.
        Returns [..., ffn]."""
        self._attrs["inputs"] = [x] + ([rrms] if rrms is not None else [])
        self._attrs["has_rrms"] = rrms is not None
        self._set_depth()
        x_shape = x._attrs["shape"]
        two_ffn = x_shape[-1]._attrs["values"][0]
        assert two_ffn % 2 == 0, "swiglu: last dim must be 2*ffn"
        from aitemplate.compiler.base import IntImm

        out_shape = list(x_shape[:-1]) + [IntImm(two_ffn // 2)]
        output = Tensor(out_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
