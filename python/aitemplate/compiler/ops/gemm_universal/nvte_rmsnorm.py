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

    def __init__(self, eps=1e-6, fp8=False) -> None:
        super().__init__()
        self._attrs["op"] = "nvte_rmsnorm"
        self._attrs["has_profiler"] = False
        self._attrs["eps"] = float(eps)
        # fp8: emit the normed output Z as E4M3 (quantized in the norm epilogue with a
        # baked z_scale) + fill amax, so a following fp8 nvte_gemm reads it directly (the
        # cast-fusion). See docs/te_fp8_impl_plan.md.
        self._attrs["fp8"] = bool(fp8)

    def __call__(self, x: Tensor, gamma: Tensor, z_scale=None) -> Tensor:
        """x: [..., C] activations, gamma: [C] weight. Returns [..., C].

        fp8: z_scale is a 1-elem fp32 constant (= 448/act_amax) used to quantize Z to
        E4M3; the output is a float8_e4m3 (1-byte) tensor the next fp8 nvte_gemm consumes.
        """
        if self._attrs["fp8"]:
            assert z_scale is not None, "nvte_rmsnorm(fp8=True) needs z_scale"
            self._attrs["inputs"] = [x, gamma, z_scale]
        else:
            self._attrs["inputs"] = [x, gamma]
        self._set_depth()
        # workspace = [fp8: amax fp32 4B + scale_inv fp32 4B] + rsigma [rows] fp32 (per-row
        # 1/rms) + a tiny nvte scratch. rows = product of leading dims; use their max for a
        # fixed prealloc. TE requires an FP8 output tensor's scale_inv be allocated (it
        # writes 1/scale there); we don't consume it (the gemm has its own act_scale_inv).
        max_rows = 1
        for d in x._attrs["shape"][:-1]:
            max_rows *= d._attrs["values"][-1]
        self._attrs["workspace"] = (
            max_rows * 4 + 4096 + (8 if self._attrs["fp8"] else 0)
        )
        out_dtype = "float8_e4m3" if self._attrs["fp8"] else x._attrs["dtype"]
        output = Tensor(list(x._attrs["shape"]), src_ops={self}, dtype=out_dtype)
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"], "fp8": self._attrs["fp8"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
