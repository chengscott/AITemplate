#  Dynamic per-row (per-token) fp8 activation quantization.
#
#  Given x [.., K] (f16), for each row r:
#     amax  = max_k |x[r,k]|
#     scale = amax / 448          (448 = e4m3 max finite magnitude)
#     xq[r,k] = to_e4m3( x[r,k] / scale )     -> float8_e4m3, same shape as x
#     scale[r]                                 -> float32, shape [.., 1]
#  so that  x ~= xq * scale  (dequantized downstream by dequant_fp8).
#
#  One warp per row, 128-bit vectorized loads and an fp32 warp reduction -- the exact
#  shape/cost of ops.rms_reduce. K must be a multiple of 8 (128-bit half loads).
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class quantize_to_fp8(Operator):
    """x [..,K] f16 -> (xq [..,K] e4m3, scale [..,1] f32) with per-row dynamic scaling."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "quantize_to_fp8"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor):
        C = x._attrs["shape"][-1]._attrs["values"][0]
        assert C % 8 == 0, f"quantize_to_fp8: last dim (C={C}) must be a multiple of 8"
        self._attrs["inputs"] = [x]
        self._set_depth()
        xq = Tensor(
            list(x._attrs["shape"]), src_ops={self}, dtype="float8_e4m3"
        )
        scale = Tensor(
            list(x._attrs["shape"][:-1]) + [IntImm(1)],
            src_ops={self},
            dtype="float32",
        )
        self._attrs["outputs"] = [xq, scale]
        return xq, scale

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
