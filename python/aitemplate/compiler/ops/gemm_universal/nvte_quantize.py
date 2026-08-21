#  TransformerEngine fp16 -> FP8(E4M3) cast as an AITemplate custom op.
#
#  Y = quantize(X) via nvte_quantize from libtransformer_engine: Y = X * scale, cast to
#  E4M3, with amax/scale_inv written by TE. Used to feed the fp8 nvte_gemm from a NON-
#  norm-fed activation (attention out -> o, silu*up -> fc2, relu(norm) -> down/up); the
#  norm-fed qkv/fc1 instead get their FP8 activation cast-fused into nvte_rmsnorm.
#  Static per-tensor scaling: `scale` (=448/act_amax) is a baked constant; the matching
#  nvte_gemm reads Y with act_scale_inv (=1/scale). See docs/te_fp8_impl_plan.md.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class nvte_quantize(Operator):
    """Y = quantize(X) to E4M3 using TransformerEngine's nvte_quantize."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "nvte_quantize"
        self._attrs["has_profiler"] = False
        # amax (4B) + scale_inv (4B): TE writes both for the FP8 output tensor and asserts
        # scale_inv is allocated; neither is read back (the gemm has its own act_scale_inv).
        self._attrs["workspace"] = 8

    def __call__(self, x: Tensor, scale: Tensor) -> Tensor:
        """x: [..., C] fp16 activation, scale: [1] fp32 (=448/act_amax). Returns E4M3."""
        self._attrs["inputs"] = [x, scale]
        self._set_depth()
        output = Tensor(
            list(x._attrs["shape"]), src_ops={self}, dtype="float8_e4m3"
        )
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
