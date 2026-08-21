#  Fused learnable-2D RoPE as an AITemplate op (one kernel; TE fused_rope in spirit).
#
#  out = rotate(X, cos, sin) on interleaved pairs (2p, 2p+1): out[2p]=x0*cos - x1*sin,
#  out[2p+1]=x0*sin + x1*cos. X: [B,S,H,D], cos/sin: [1,S,H,P] (P=D/2, baked at export) ->
#  out: [B,S,H,D]. Replaces the split + mul/sub/add + concatenate chain with one kernel.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class rope2d(Operator):
    """out = learnable-2D RoPE(X, cos, sin), interleaved pairs; X/out [B,S,H,D]."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "rope2d"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor, cosb: Tensor, sinb: Tensor) -> Tensor:
        """x: [B,S,H,D]; cos/sin: [1,S,H,P] (P=D/2). Returns [B,S,H,D]."""
        self._attrs["inputs"] = [x, cosb, sinb]
        self._set_depth()
        xshape = x._attrs["shape"]  # [B,S,H,D]
        S = xshape[1]._attrs["values"][0]
        H = xshape[2]._attrs["values"][0]
        D = xshape[3]._attrs["values"][0]
        assert D % 2 == 0, "rope: head_dim must be even"
        self._attrs["sh"] = S * H
        self._attrs["p"] = D // 2
        output = Tensor(list(xshape), src_ops={self}, dtype=x._attrs["dtype"])
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
