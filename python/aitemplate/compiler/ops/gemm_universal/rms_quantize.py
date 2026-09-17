#  Fused rms_reduce + quantize_to_fp8: reads the RAW activation x [..,C] ONCE and emits both
#  the per-token rrms scalar (the rmsnorm-prologue reduction) AND the e4m3 quantization of x
#  (per-row absmax scale). Replaces a separate rms_reduce (1 read of x) + quantize_to_fp8
#  (2 reads of x) in the fp8 attention/ffn sub-blocks, where the qkv/fc1 gemm runs on raw x
#  (gamma folded into its weight) and rrms is applied downstream in unpack_rope / swiglu.
#  x [..,C] f16 -> (xq [..,C] e4m3, scale [..,1] f32, rrms [..,1] f16). C % 8 == 0.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class rms_quantize(Operator):
    """(xq e4m3, scale f32, rrms f16) = fused(rms_reduce(x), quantize_to_fp8(x)); x [..,C]."""

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self._attrs["op"] = "rms_quantize"
        self._attrs["has_profiler"] = False
        self._attrs["eps"] = float(eps)

    def __call__(self, x: Tensor):
        C = x._attrs["shape"][-1]._attrs["values"][0]
        assert C % 8 == 0, f"rms_quantize: last dim (C={C}) must be a multiple of 8"
        self._attrs["inputs"] = [x]
        self._set_depth()
        xq = Tensor(list(x._attrs["shape"]), src_ops={self}, dtype="float8_e4m3")
        row_shape = list(x._attrs["shape"][:-1]) + [IntImm(1)]
        scale = Tensor(row_shape, src_ops={self}, dtype="float32")
        rrms = Tensor(row_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [xq, scale, rrms]
        return xq, scale, rrms

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
