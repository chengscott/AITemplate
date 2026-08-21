#  TransformerEngine cuBLAS/FP8 GEMM as an AITemplate custom op.
#
#  Y = X @ W^T  via nvte_cublas_gemm from libtransformer_engine (the same C kernel
#  te.Linear calls). X: [..., K], W: [N, K] -> Y: [..., N]. fp16 today; the FP8 path
#  (amax/scale/scale_inv + FP8 dtype) is a config extension for SM90+ (H200).
#
#  Modeled on the flash_attention external-kernel op. See nvte_gemm backend codegen
#  (backend/cuda/gemm_universal/nvte_gemm.py) for the emitted C++.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0223

# cuBLASLt workspace (bytes). 32MB is the size TE's PyTorch path uses on Hopper.
_WORKSPACE_BYTES = 32 * 1024 * 1024


class nvte_gemm(Operator):
    """Y = X @ W^T using TransformerEngine's nvte_cublas_gemm."""

    def __init__(self, fp8=False) -> None:
        super().__init__()
        self._attrs["op"] = "nvte_gemm"
        self._attrs["has_profiler"] = False
        self._attrs["workspace"] = _WORKSPACE_BYTES
        # fp8: A (weight) and B (activation) are E4M3 with baked per-tensor scale_inv,
        # split-accumulator on; D stays fp16. Static per-tensor scaling (scales are
        # constants) -> no runtime amax bookkeeping. See docs/te_fp8_impl_plan.md.
        self._attrs["fp8"] = bool(fp8)

    def _infer_shapes(self, x: Tensor, w: Tensor):
        x_shape = x._attrs["shape"]
        w_shape = w._attrs["shape"]
        assert len(w_shape) == 2, "nvte_gemm weight must be 2D [N, K]"
        # contraction dim K must match (last dim of x, last dim of w)
        assert (
            x_shape[-1]._attrs["values"] == w_shape[1]._attrs["values"]
        ), "nvte_gemm: X last dim (K) must match W dim1 (K)"
        return list(x_shape[:-1]) + [w_shape[0]]

    def __call__(
        self, x: Tensor, w: Tensor, act_scale_inv=None, w_scale_inv=None
    ) -> Tensor:
        """x: [..., K] activations, w: [N, K] weight. Returns [..., N].

        fp8: x is a float8_e4m3 (1-byte) tensor, w is float8_e4m3, and act_scale_inv /
        w_scale_inv are 1-elem fp32 constants that descale the gemm. The output is fp16.
        """
        if self._attrs["fp8"]:
            assert (
                act_scale_inv is not None and w_scale_inv is not None
            ), "nvte_gemm(fp8=True) needs act_scale_inv and w_scale_inv"
            self._attrs["inputs"] = [x, w, act_scale_inv, w_scale_inv]
        else:
            self._attrs["inputs"] = [x, w]
        self._set_depth()
        # fp8 activation carries a 1-byte dtype; force the gemm output back to fp16.
        out_dtype = "float16" if self._attrs["fp8"] else x._attrs["dtype"]
        output = Tensor(self._infer_shapes(x, w), src_ops={self}, dtype=out_dtype)
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"fp8": self._attrs["fp8"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
