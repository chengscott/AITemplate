#  Single-pass fp8 quantization using a PRECOMPUTED amax (no reduction). The amax of this
#  activation was already computed in the producing conv's epilogue (conv2d_fp8 emit_amax),
#  so quantize is just a pointwise divide:
#     scale = amax[0]/448 ; xq = to_e4m3(x / scale)
#  x [..] f16 + amax [1] f32 -> xq [..] e4m3 (same shape) + scale [1] f32.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class quantize_to_fp8_amax(Operator):
    """x f16 + amax [1] -> (xq e4m3 same-shape, scale [1]) single-pass (no reduction)."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "quantize_to_fp8_amax"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor, amax: Tensor):
        self._attrs["inputs"] = [x, amax]
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
