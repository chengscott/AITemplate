#  Native RMSNorm as an AITemplate op (ports te.rmsnorm's fused kernel into AIT).
#
#  Z = RMSNorm(X) * gamma via a single inline CUDA kernel (backend codegen in
#  backend/cuda/gemm_universal/rmsnorm.py) -- NOT the libtransformer_engine wrapper
#  (nvte_rmsnorm) and NOT an elementwise reduce_mean decomposition. One fused kernel,
#  compiled in the same nvcc pass as the rest of the cutlass graph. X: [..., C],
#  gamma: [C] -> Z: [..., C]. fp16 today.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class rmsnorm(Operator):
    """Z = RMSNorm(X) * gamma, single-pass fused kernel (native port of te.rmsnorm_fwd)."""

    def __init__(self, eps=1e-6, relu=False, fp8_out=False) -> None:
        super().__init__()
        self._attrs["op"] = "rmsnorm"
        self._attrs["has_profiler"] = False
        self._attrs["eps"] = float(eps)
        # fuse a relu on the output (RMSNorm -> relu, e.g. nbt norm_p/norm_q) so the relu
        # isn't a separate elementwise kernel.
        self._attrs["relu"] = bool(relu)
        # fp8_out: also quantize the (normed+relu'd) output to e4m3 per-row in-kernel, so a
        # downstream fp8 gemm needs no separate quantize. Returns (xq e4m3, scale f32[..,1]).
        # Requires C <= 256 (each warp lane holds its uint4 chunk in a register).
        self._attrs["fp8_out"] = bool(fp8_out)

    def __call__(self, x: Tensor, gamma: Tensor):
        """x: [..., C] activations, gamma: [C] weight. Returns [..., C] (or (xq e4m3,
        scale) when fp8_out). C must be a multiple of 8 (uint4 vectorized loads)."""
        C = x._attrs["shape"][-1]._attrs["values"][0]
        assert C % 8 == 0, f"rmsnorm: last dim (C={C}) must be a multiple of 8"
        self._attrs["inputs"] = [x, gamma]
        self._set_depth()
        if self._attrs["fp8_out"]:
            from aitemplate.compiler.base import IntImm

            assert C <= 256, f"rmsnorm fp8_out needs C <= 256, got {C}"
            xq = Tensor(list(x._attrs["shape"]), src_ops={self}, dtype="float8_e4m3")
            scale = Tensor(
                list(x._attrs["shape"][:-1]) + [IntImm(1)],
                src_ops={self},
                dtype="float32",
            )
            self._attrs["outputs"] = [xq, scale]
            return xq, scale
        output = Tensor(
            list(x._attrs["shape"]), src_ops={self}, dtype=x._attrs["dtype"]
        )
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {
            "eps": self._attrs["eps"],
            "relu": self._attrs["relu"],
            "fp8_out": self._attrs["fp8_out"],
        }

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
