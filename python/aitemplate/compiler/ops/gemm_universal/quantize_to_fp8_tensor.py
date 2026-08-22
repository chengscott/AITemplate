#  Dynamic PER-TENSOR fp8 activation quantization (for the fp8 implicit-gemm conv path).
#  Unlike quantize_to_fp8 (per-row), a conv output pixel gathers from many input pixels, so
#  a per-row activation scale doesn't fold into the epilogue -- a single global scalar does.
#
#    x [..] f16 -> xq [..] e4m3 (same shape) + scale [1] f32 = amax(|x|)/448
#
#  so x ~= xq * scale (dequantized by dequant_fp8 with scalar_scale=True).
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class quantize_to_fp8_tensor(Operator):
    """x f16 -> (xq e4m3 same-shape, scale f32 [1]) with one dynamic per-tensor scale."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "quantize_to_fp8_tensor"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor):
        self._attrs["inputs"] = [x]
        self._set_depth()
        xq = Tensor(list(x._attrs["shape"]), src_ops={self}, dtype="float8_e4m3")
        scale = Tensor([IntImm(1)], src_ops={self}, dtype="float32")
        self._attrs["outputs"] = [xq, scale]
        return xq, scale

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
