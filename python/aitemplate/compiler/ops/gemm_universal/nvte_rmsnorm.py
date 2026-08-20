#  TransformerEngine RMSNorm as an AITemplate custom op.
#
#  Z = RMSNorm(X) * gamma  via nvte_rmsnorm_fwd from libtransformer_engine (the fused
#  kernel te.RMSNorm calls). X: [..., C], gamma: [C] -> Z: [..., C]. fp16 today; the FP8
#  path (emit FP8 Z + amax for a following FP8 gemm) is a config extension for SM90+.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor


class nvte_rmsnorm(Operator):
    """Z = RMSNorm(X) * gamma using TransformerEngine's nvte_rmsnorm_fwd."""

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self._attrs["op"] = "nvte_rmsnorm"
        self._attrs["has_profiler"] = False
        self._attrs["eps"] = float(eps)

    def __call__(self, x: Tensor, gamma: Tensor) -> Tensor:
        """x: [..., C] activations, gamma: [C] weight. Returns [..., C]."""
        self._attrs["inputs"] = [x, gamma]
        self._set_depth()
        # workspace = rsigma [rows] fp32 (per-row 1/rms) + a tiny nvte scratch. rows is
        # the product of leading dims; use their max for a fixed pre-alloc.
        max_rows = 1
        for d in x._attrs["shape"][:-1]:
            max_rows *= d._attrs["values"][-1]
        self._attrs["workspace"] = max_rows * 4 + 4096
        output = Tensor(
            list(x._attrs["shape"]), src_ops={self}, dtype=x._attrs["dtype"]
        )
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
