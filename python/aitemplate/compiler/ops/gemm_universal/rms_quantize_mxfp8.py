#  MXFP8 producer fusion for the fp8 qkv/fc1 sub-blocks: reads the raw activation x [..,C] ONCE
#  and emits (xq e4m3, sfa swizzled ue8m0 per-32-K block scales, rrms per-token) -- the mxfp8
#  analog of rms_quantize. The downstream gemm_rcr_mxfp8(a_scale=sfa) then skips its internal
#  quantize. sfa is a flat swizzled buffer sized rows*ceil(nkb/4)*4 + ceil(nkb/4)*512 (an upper
#  bound on the cuBLAS d-block-scaling layout size). C % 32 == 0.
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221,W0223


class rms_quantize_mxfp8(Operator):
    """(xq e4m3, sfa ue8m0 swizzled, rrms f16) = fused rms_reduce + mxfp8-quantize(x)."""

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self._attrs["op"] = "rms_quantize_mxfp8"
        self._attrs["has_profiler"] = False
        self._attrs["eps"] = float(eps)

    def __call__(self, x: Tensor):
        C = x._attrs["shape"][-1]._attrs["values"][0]
        assert C % 32 == 0, f"rms_quantize_mxfp8 needs C % 32 == 0, got {C}"
        nkb = C // 32
        ntx = (nkb + 3) // 4
        self._attrs["inputs"] = [x]
        self._set_depth()
        xq = Tensor(list(x._attrs["shape"]), src_ops={self}, dtype="float8_e4m3")
        rows = x._attrs["shape"][0]
        for d in x._attrs["shape"][1:-1]:
            rows = rows * d
        sfa_dim = rows * (ntx * 4) + ntx * 512  # upper-bound swizzled SFA byte count
        sfa = Tensor([sfa_dim], src_ops={self}, dtype="float8_e4m3")  # ue8m0 1-byte carrier
        rrms = Tensor(
            list(x._attrs["shape"][:-1]) + [IntImm(1)], src_ops={self}, dtype=x._attrs["dtype"]
        )
        self._attrs["outputs"] = [xq, sfa, rrms]
        return xq, sfa, rrms

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        return registry.get(func_key)(self._attrs)
