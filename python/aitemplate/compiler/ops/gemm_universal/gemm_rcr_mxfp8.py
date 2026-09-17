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

    def __call__(self, a, b, b_scale, residual=None, a_scale=None):
        # a [M,K]: f16 (quantized to e4m3+SFA internally) when a_scale is None, else already
        # e4m3 with a_scale the swizzled ue8m0 SFA from a producer (rms_quantize_mxfp8 /
        # rmsnorm mxfp8_out / swiglu mxfp8_out). b [N,K] e4m3, b_scale ue8m0 SFB (swizzled,
        # baked at export). K % 32 == 0 (SFVecSize).
        K = a._attrs["shape"][-1]._attrs["values"][0]
        Kb = b._attrs["shape"][1]._attrs["values"][0]
        Nn = b._attrs["shape"][0]._attrs["values"][0]
        assert K == Kb, f"gemm_rcr_mxfp8 K mismatch {K} vs {Kb}"
        assert K % 32 == 0, f"gemm_rcr_mxfp8 needs K % 32 == 0 (SFVecSize), got {K}"
        self._attrs["N"], self._attrs["K"] = Nn, K
        self._attrs["has_prequant"] = a_scale is not None
        self._attrs["has_residual"] = residual is not None
        inputs = [a, b, b_scale]
        if a_scale is not None:
            inputs.append(a_scale)
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
