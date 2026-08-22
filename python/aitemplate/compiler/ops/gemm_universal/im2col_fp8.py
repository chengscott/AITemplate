#  Fused im2col + dynamic fp8 quantization for the fp8 conv path (SM90 has no fp8
#  implicit-gemm conv kernels, so convs run as im2col -> fp8 gemm).
#
#  x NHWC [B,H,W,Cin] -> for each output position (b,oh,ow) gather the kh*kw*Cin window
#  (zero-padded), and emit a per-row-quantized e4m3 row + its f32 scale:
#     xq    [B, OH*OW, K]  e4m3   (K = kh*kw*Cin, column order (ky,kx,c))
#     scale [B, OH*OW, 1]  f32    (row absmax / 448)
#  so that  window ~= xq * scale.  The reshaped conv weight [Cout, K] (order (ky,kx,c),
#  matching NHWC [Cout,kh,kw,Cin]) is the gemm's B operand. 1x1 convs are the kh=kw=1
#  special case (K=Cin, no padding). Same-pad stride-1 -> OH=H, OW=W.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class im2col_fp8(Operator):
    """x[B,H,W,Cin] f16 -> (xq[B,OH*OW,kh*kw*Cin] e4m3, scale[B,OH*OW,1] f32)."""

    def __init__(self, kh, kw, stride=1, pad=None) -> None:
        super().__init__()
        self._attrs["op"] = "im2col_fp8"
        self._attrs["has_profiler"] = False
        self._attrs["kh"] = int(kh)
        self._attrs["kw"] = int(kw)
        self._attrs["stride"] = int(stride)
        # default: same padding for stride 1 (pad = (k-1)/2)
        self._attrs["pad"] = int(pad) if pad is not None else (int(kh) - 1) // 2

    def __call__(self, x: Tensor):
        B, H, W, Cin = x._attrs["shape"]
        Hs, Ws, Cs = (d._attrs["values"][0] for d in (H, W, Cin))
        kh, kw, st, pad = (
            self._attrs["kh"],
            self._attrs["kw"],
            self._attrs["stride"],
            self._attrs["pad"],
        )
        OH = (Hs + 2 * pad - kh) // st + 1
        OW = (Ws + 2 * pad - kw) // st + 1
        K = kh * kw * Cs
        assert K % 8 == 0, f"im2col_fp8: K={K} (kh*kw*Cin) must be a multiple of 8"
        self._attrs["OH"], self._attrs["OW"], self._attrs["Cin"] = OH, OW, Cs
        self._attrs["H"], self._attrs["W"] = Hs, Ws
        self._attrs["inputs"] = [x]
        self._set_depth()
        xq = Tensor([B, IntImm(OH * OW), IntImm(K)], src_ops={self}, dtype="float8_e4m3")
        scale = Tensor([B, IntImm(OH * OW), IntImm(1)], src_ops={self}, dtype="float32")
        self._attrs["outputs"] = [xq, scale]
        return xq, scale

    def _get_op_attributes(self):
        return {
            "kh": self._attrs["kh"],
            "kw": self._attrs["kw"],
            "stride": self._attrs["stride"],
            "pad": self._attrs["pad"],
        }

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
