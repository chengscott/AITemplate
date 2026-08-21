#  Per-row RMS reduction as an AITemplate op: the "prologue" half of RMSNorm.
#
#  rrms[r] = rsqrt(mean_k(x[r,k]^2) + eps)  for x: [..., C] -> rrms: [..., 1].
#  Used by the rmsnorm-prologue fusion: instead of writing the full normalized activation
#  z = x * rrms * gamma [.., C] (a whole [rows,C] round-trip), we (1) fold gamma into the
#  next gemm's weight at export, (2) precompute just this per-row rrms scalar here, and (3)
#  apply rrms in the downstream kernel that already streams the gemm output (unpack_rope /
#  swiglu). Net: the [rows,C] write of the normalized activation disappears. fp16 in,
#  fp16 rrms out (fp32 reduction).
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class rms_reduce(Operator):
    """rrms[r] = rsqrt(mean(x[r,:]^2) + eps); x [..,C] -> rrms [..,1]. C % 8 == 0."""

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self._attrs["op"] = "rms_reduce"
        self._attrs["has_profiler"] = False
        self._attrs["eps"] = float(eps)

    def __call__(self, x: Tensor) -> Tensor:
        C = x._attrs["shape"][-1]._attrs["values"][0]
        assert C % 8 == 0, f"rms_reduce: last dim (C={C}) must be a multiple of 8"
        self._attrs["inputs"] = [x]
        self._set_depth()
        out_shape = list(x._attrs["shape"][:-1]) + [IntImm(1)]
        output = Tensor(out_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
