#  Fused fp8 (e4m3) gemm on SM90 (RCR) via a hand-emitted CUTLASS 3x GemmUniversalAdapter
#  whose epilogue folds dequant + bias + residual:
#     D_f16 = (scale_x*scale_w) * (A @ B^T) + bias[n] + residual   (writes f16 directly)
#  A [M,K] e4m3 (RowMajor), B [N,K] e4m3 (ColMajor), scale_x/scale_w [1] f32 (per-tensor),
#  bias [N] f16, residual [M,N] f16 (optional). No f32 acc round-trip / separate dequant.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class gemm_rcr_fp8_fused(Operator):
    """(A e4m3, B e4m3) -> D f16 = scale*(A@B^T) + bias + residual, fused fp8 gemm."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "gemm_rcr_fp8_fused"
        self._attrs["has_profiler"] = False

    def __call__(self, a, b, scale_x, scale_w, residual=None):
        # a [M,K] e4m3, b [N,K] e4m3, scale_x [M,1] f32 (per-row/per-token from
        # quantize_to_fp8), scale_w [1] f32. Bias-free (linears emit zero bias).
        K = a._attrs["shape"][-1]._attrs["values"][0]
        Kb = b._attrs["shape"][1]._attrs["values"][0]
        Nn = b._attrs["shape"][0]._attrs["values"][0]
        assert K == Kb, f"gemm_rcr_fp8_fused K mismatch {K} vs {Kb}"
        assert K % 16 == 0, f"gemm_rcr_fp8_fused needs K % 16 == 0 (e4m3 TMA), got {K}"
        self._attrs["N"], self._attrs["K"] = Nn, K
        inputs = [a, b, scale_x, scale_w]
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
