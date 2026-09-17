#  SM100 MXFP8 (block-scaled e4m3) gemm (RCR): A/B are e4m3 data + per-32-element ue8m0 block
#  scales (SFA activation, SFB weight, both in the swizzled cuBLAS layout). Block scales are
#  applied inside the tcgen05 UMMA, so D_f16 = (A@B^T) + residual (bias-free trunk). No per-row
#  alpha / dequant kernel. A [M,K] e4m3, B [N,K] e4m3, residual [M,N] f16 (optional).
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class gemm_rcr_mxfp8(Operator):
    """(A e4m3+SFA, B e4m3+SFB) -> D f16 = (A@B^T) + residual, SM100 block-scaled gemm."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "gemm_rcr_mxfp8"
        self._attrs["has_profiler"] = False

    def __call__(self, a, sfa, b, sfb, residual=None):
        # a [M,K] e4m3, sfa ue8m0 (swizzled), b [N,K] e4m3, sfb ue8m0 (swizzled).
        K = a._attrs["shape"][-1]._attrs["values"][0]
        Kb = b._attrs["shape"][1]._attrs["values"][0]
        Nn = b._attrs["shape"][0]._attrs["values"][0]
        assert K == Kb, f"gemm_rcr_mxfp8 K mismatch {K} vs {Kb}"
        assert K % 32 == 0, f"gemm_rcr_mxfp8 needs K % 32 == 0 (SFVecSize), got {K}"
        self._attrs["N"], self._attrs["K"] = Nn, K
        inputs = [a, sfa, b, sfb]
        if residual is not None:
            inputs.append(residual)
        self._attrs["has_residual"] = residual is not None
        self._attrs["inputs"] = inputs
        self._set_depth()
        out = Tensor(
            list(a._attrs["shape"][:-1]) + [IntImm(Nn)], src_ops={self}, dtype="float16"
        )
        self._attrs["outputs"] = [out]
        return out

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
