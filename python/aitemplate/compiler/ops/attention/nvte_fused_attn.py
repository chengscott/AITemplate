#  TransformerEngine fused attention (cuDNN) as an AITemplate custom op.
#
#  O = softmax(Q K^T * scale) V  via nvte_fused_attn_fwd, BSHD/no-mask/inference.
#  Q,K,V: [B,S,H,D] -> O: [B,S,H,D]. fp16 today; FP8 attention is a config extension
#  for SM90+ (H200). Mirrors te.DotProductAttention's fwd (used in training).
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor


class nvte_fused_attn(Operator):
    """O = Attention(Q,K,V) via TransformerEngine's nvte_fused_attn_fwd (BSHD, no mask)."""

    def __init__(self, scale, seq_len) -> None:
        super().__init__()
        self._attrs["op"] = "nvte_fused_attn"
        self._attrs["has_profiler"] = False
        self._attrs["scale"] = float(scale)
        self._attrs["seq_len"] = int(seq_len)

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """q,k,v: [B,S,H,D] (post-RoPE). Returns O: [B,S,H,D]."""
        self._attrs["inputs"] = [q, k, v]
        self._set_depth()
        qshape = q._attrs["shape"]  # [B, S, H, D]
        B = qshape[0]
        H = qshape[2]._attrs["values"][0]
        D = qshape[3]._attrs["values"][0]
        S = self._attrs["seq_len"]
        max_b = B._attrs["values"][-1]
        self._attrs["max_batch"] = max_b
        self._attrs["heads"] = H
        self._attrs["head_dim"] = D
        # global_workspace_ layout (fixed max): cu_seqlens_q/kv [(maxB+1) int32]x2,
        # rng_in [2 int64], aux region (softmax-stats [maxB,H,S] fp32 + rng_out [2] +
        # slack), nvte scratch. Reserve generously.
        self._attrs["workspace"] = (
            2 * (max_b + 1) * 4 + 16 + max_b * H * S * 4 + 4096
        )
        output = Tensor(list(qshape), src_ops={self}, dtype=q._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"scale": self._attrs["scale"], "seq_len": self._attrs["seq_len"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
