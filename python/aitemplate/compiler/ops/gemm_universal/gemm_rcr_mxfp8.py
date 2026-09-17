#  SM100 MXFP8 (block-scaled e4m3) gemm (RCR) with the ACTIVATION quantize folded into the op:
#  takes the f16 activation A [M,K] and, inside the emitted function, quantizes it to e4m3 +
#  per-32-element ue8m0 block scales (SFA, swizzled) into static workspace buffers, then runs the
#  tcgen05 block-scaled UMMA against the e4m3 weight B [N,K] + its baked swizzled ue8m0 scales
#  (SFB). Block scales apply inside the MMA -> D_f16 = (A@B^T) + residual (bias-free trunk).
#  Keeping SFA a static device workspace (not a graph tensor) avoids a dynamic-M swizzled shape.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class gemm_rcr_mxfp8(Operator):
    """(A f16, B e4m3, B_sf ue8m0) -> D f16 = (A@B^T)+residual; SM100 block-scaled gemm."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "gemm_rcr_mxfp8"
        self._attrs["has_profiler"] = False

    def __call__(self, a, b, b_scale, residual=None, a_scale=None,
                 norm="none", gamma=None, relu=False, eps=1e-6, rrms=None):
        # a [M,K]: f16 quantized to e4m3+SFA internally (a_scale None). norm='rmsnorm' folds
        # RMSNorm(a)*gamma (+relu) into that internal quantize; norm='swiglu' folds SwiGLU
        # (a is the packed fc1 [M,2*ffn]; z=silu(rrms*gate)*(rrms*up)) into it -- both
        # mega-fused (no separate norm/swiglu kernel + f16 round-trip). The SFA stays a STATIC
        # workspace (no dynamic SF graph tensor). b [N,K] e4m3, b_scale ue8m0 SFB (swizzled,
        # baked at export). K % 32 == 0. In swiglu mode the gemm K = ffn = a's last dim // 2.
        A_last = a._attrs["shape"][-1]._attrs["values"][0]
        K = A_last // 2 if norm == "swiglu" else A_last
        Kb = b._attrs["shape"][1]._attrs["values"][0]
        Nn = b._attrs["shape"][0]._attrs["values"][0]
        assert K == Kb, f"gemm_rcr_mxfp8 K mismatch {K} vs {Kb}"
        assert K % 32 == 0, f"gemm_rcr_mxfp8 needs K % 32 == 0 (SFVecSize), got {K}"
        assert norm in ("none", "rmsnorm", "swiglu"), f"gemm_rcr_mxfp8 unknown norm {norm}"
        self._attrs["N"], self._attrs["K"] = Nn, K
        self._attrs["has_prequant"] = a_scale is not None
        self._attrs["has_residual"] = residual is not None
        self._attrs["norm"] = norm
        self._attrs["eps"] = float(eps)
        self._attrs["relu"] = bool(relu)
        # inputs: [a, b, b_scale] + [a_scale?] + [gamma? (rmsnorm)] + [rrms? (swiglu)] + [residual?]
        inputs = [a, b, b_scale]
        if a_scale is not None:
            inputs.append(a_scale)
        if norm == "rmsnorm":
            assert gamma is not None, "gemm_rcr_mxfp8 norm='rmsnorm' needs gamma"
            inputs.append(gamma)
        if norm == "swiglu":
            assert rrms is not None, "gemm_rcr_mxfp8 norm='swiglu' needs rrms"
            inputs.append(rrms)
        if residual is not None:
            inputs.append(residual)
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
