#  Fused qkv-unpack + RoPE(q,k) as one AITemplate op.
#
#  Input: packed qkv [B, S, 3*dim] (the fused-qkv gemm output, laid out [.,3,H,D]) plus
#  baked cos/sin [1,S,H,P] (P=D/2). Outputs three CONTIGUOUS tensors q,k,v [B,S,H,D] that
#  FlashAttention consumes directly: q,k are RoPE-rotated, v is copied. This folds the
#  generic 3-way split AND the two rope2d passes into a single read-once/write-once kernel
#  (the split's write is unavoidable to hand FA contiguous q/k/v, but the rope memory
#  passes disappear). Mirrors AitNBTMHSA: split -> reshape -> rope(q),rope(k) -> attention.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class unpack_rope(Operator):
    """qkv [B,S,3*dim] + cos/sin -> (q,k,v) [B,S,H,D]; q,k RoPE-rotated, v copied."""

    def __init__(self, heads) -> None:
        super().__init__()
        self._attrs["op"] = "unpack_rope"
        self._attrs["has_profiler"] = False
        self._attrs["heads"] = int(heads)

    def __call__(self, qkv: Tensor, cosb: Tensor, sinb: Tensor, rrms: Tensor = None):
        """qkv: [B,S,3*dim] (3*dim = 3*H*D); cos/sin: [1,S,H,P] (P=D/2). Optional rrms
        [B,S,1] (rmsnorm-prologue): each token's q/k/v is scaled by rrms before rope/copy.
        Returns (q, k, v), each [B,S,H,D] contiguous."""
        self._attrs["inputs"] = [qkv, cosb, sinb] + ([rrms] if rrms is not None else [])
        self._attrs["has_rrms"] = rrms is not None
        self._set_depth()
        qshape = qkv._attrs["shape"]  # [B, S, 3*dim]
        B = qshape[0]
        S = qshape[1]._attrs["values"][0]
        three_dim = qshape[2]._attrs["values"][0]
        assert three_dim % 3 == 0, "unpack_rope: last dim must be 3*dim"
        dim = three_dim // 3
        H = self._attrs["heads"]
        assert dim % H == 0, "unpack_rope: dim must be divisible by heads"
        D = dim // H
        assert D % 2 == 0, "unpack_rope: head_dim must be even"
        self._attrs["sh"] = S * H
        self._attrs["p"] = D // 2
        self._attrs["dim"] = dim
        self._attrs["head_dim"] = D
        from aitemplate.compiler.base import IntImm

        oshape = [B, IntImm(S), IntImm(H), IntImm(D)]
        outs = [Tensor(list(oshape), src_ops={self}, dtype=qkv._attrs["dtype"]) for _ in range(3)]
        self._attrs["outputs"] = outs
        self._attrs["output_masks"] = [True, True, True]
        return tuple(outs)

    def _get_op_attributes(self):
        return {"heads": self._attrs["heads"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
