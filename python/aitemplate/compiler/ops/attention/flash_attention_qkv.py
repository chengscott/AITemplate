#  FA4-backed fused attention taking SEPARATE q, k, v (no packed-qkv concatenate).
#
#  O = softmax(scale * Q K^T) V via the FlashAttention-4 (CuTeDSL) kernel, with Q, K, V, O
#  each a contiguous [B, S, H, D] tensor. This is the fused-attention op (the
#  DotProductAttention / nvte_fused_attn role) with cuDNN replaced by FA4, and -- unlike
#  ops.flash_attention's packed 5D path -- it feeds q/k/v directly, dropping the concat
#  packing copy. scale = head_dim**-0.5 is baked into the AOT kernel. fp16, head_dim in
#  {8,16,32,64,128}, non-causal/no-mask.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class flash_attention_qkv(Operator):
    """O = FA4(Q,K,V), Q/K/V/O contiguous [B,S,H,D]; scale=head_dim**-0.5 baked."""

    def __init__(self, seq_len, causal=False) -> None:
        super().__init__()
        self._attrs["op"] = "flash_attention_qkv"
        self._attrs["has_profiler"] = False
        self._attrs["seq_len"] = int(seq_len)
        self._attrs["causal"] = bool(causal)

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """q,k,v: [B,S,H,D] (post-RoPE), contiguous. Returns O: [B,S,H,D]."""
        self._attrs["inputs"] = [q, k, v]
        self._set_depth()
        qshape = q._attrs["shape"]  # [B, S, H, D]
        B = qshape[0]
        H = qshape[2]._attrs["values"][0]
        D = qshape[3]._attrs["values"][0]
        assert D in (8, 16, 32, 64, 128), "flash_attention_qkv: head_dim in {8,16,...,128}"
        self._attrs["heads"] = H
        self._attrs["head_dim"] = D
        # softmax-lse scratch [B,H,S] fp32 (max batch); carved from global_workspace_.
        max_b = B._attrs["values"][-1]
        self._attrs["workspace"] = max_b * H * self._attrs["seq_len"] * 4
        output = Tensor(list(qshape), src_ops={self}, dtype=q._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"seq_len": self._attrs["seq_len"], "causal": self._attrs["causal"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
