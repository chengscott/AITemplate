#  fp8 (e4m3) GEMM: A[RowMajor], B[ColMajor] both float8_e4m3, accumulated in fp32
#  and written to a **float32** output (no scaling applied here). The SM90 WGMMA fp8
#  kernels cutlass already emits are selected by default_fproc via the e4m3 input dtype
#  + f32 output dtype (see backend/cuda/gemm_universal/common.py::default_fproc).
#
#  This op deliberately keeps the backend op name "gemm_rcr" so ALL of gemm_rcr's
#  codegen / profiler / config is reused verbatim -- the only difference from a normal
#  gemm_rcr is that the output tensor is float32 (via _output_dtype). Dequantization
#  (apply per-token activation scale * per-tensor weight scale, add bias, cast to f16)
#  is a separate fused op (dequant_fp8). Pairs with quantize_to_fp8 upstream.
from aitemplate.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr

# pylint: disable=C0103,W0223


class gemm_rcr_fp8(gemm_rcr):
    """C_f32 = A_e4m3 @ B_e4m3^T (fp32 accumulate). RowMajor A, ColMajor B."""

    def __init__(self):
        super().__init__()
        # NOTE: _attrs["op"] stays "gemm_rcr" (inherited) on purpose -> reuse the
        # gemm_rcr backend registry entries (config/gen_function/func_call/filter).
        # The profiler cache keys on the cutlass A/B/C element types, so this never
        # collides with a real f16 gemm_rcr of the same M/N/K.

    def _output_dtype(self) -> str:
        # fp8 A/B -> fp32 accumulator written out as float32 (higher-precision acc).
        return "float32"
