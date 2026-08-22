#  Fused fp8 (e4m3) implicit-GEMM convolution (fprop) on SM90 via CUTLASS 3x
#  ConvUniversalAdapter, with the dequant folded into the epilogue (no f32 round-trip,
#  no separate dequant kernel):
#
#    D_f16 = act( (scale_x*scale_w) * conv(xq, w) + bias[k] + residual )
#
#  xq [N,H,W,C] e4m3, w [K,R,S,C] e4m3, scale_x/scale_w [1] f32 (per-tensor), bias [K] f16,
#  residual [N,OH,OW,K] f16 (optional). Static conv geometry baked; batch N runtime.
#  fprop-only (fp8 WGMMA atoms are TN/K-major). Epilogue = LinCombPerColBiasEltAct.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class conv2d_fp8(Operator):
    """SM90 fp8 implicit-gemm conv (fprop) with fused dequant/bias/relu/residual epilogue."""

    def __init__(self, stride=1, pad=1, relu=False, emit_amax=False) -> None:
        super().__init__()
        self._attrs["op"] = "conv2d_fp8"
        self._attrs["has_profiler"] = False
        self._attrs["stride"] = int(stride)
        self._attrs["pad"] = int(pad)
        self._attrs["relu"] = bool(relu)
        # emit_amax: also compute amax(output) in the epilogue (Sm90ScalarReduction), so the
        # NEXT conv's quantize is single-pass. Adds a 2nd output (amax [1] f32).
        self._attrs["emit_amax"] = bool(emit_amax)

    def __call__(self, x, w, scale_x, scale_w, bias, residual=None):
        N, H, W, C = x._attrs["shape"]
        K, R, S, C2 = w._attrs["shape"]
        Hs, Ws, Cs = (d._attrs["values"][0] for d in (H, W, C))
        Ks, Rs, Ss = (d._attrs["values"][0] for d in (K, R, S))
        st, pad = self._attrs["stride"], self._attrs["pad"]
        OH = (Hs + 2 * pad - Rs) // st + 1
        OW = (Ws + 2 * pad - Ss) // st + 1
        assert Cs % 16 == 0 and Ks % 16 == 0, (
            f"conv2d_fp8 needs C,K % 16 == 0 (TMA e4m3), got C={Cs} K={Ks}"
        )
        self._attrs.update(dict(H=Hs, W=Ws, C=Cs, K=Ks, R=Rs, S=Ss, OH=OH, OW=OW))
        inputs = [x, w, scale_x, scale_w, bias]
        if residual is not None:
            inputs.append(residual)
        self._attrs["has_residual"] = residual is not None
        self._attrs["inputs"] = inputs
        self._set_depth()
        out = Tensor(
            [N, IntImm(OH), IntImm(OW), IntImm(Ks)], src_ops={self}, dtype="float16"
        )
        if self._attrs["emit_amax"]:
            amax = Tensor([IntImm(1)], src_ops={self}, dtype="float32")
            self._attrs["outputs"] = [out, amax]
            return out, amax
        self._attrs["outputs"] = [out]
        return out

    def _get_op_attributes(self):
        return {
            "stride": self._attrs["stride"],
            "pad": self._attrs["pad"],
            "relu": self._attrs["relu"],
            "emit_amax": self._attrs["emit_amax"],
        }

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
