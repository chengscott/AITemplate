"""CuTe attention tiles for short sequences with narrow heads."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

try:
    from cutlass.base_dsl.arch import Arch
except ModuleNotFoundError:
    from cutlass.base_dsl.enums import Arch
from .cutedsl_flash_attention_sm80 import FlashAttentionFwdSm80Aot


class FlashAttentionShortAot:
    def __init__(
        self,
        head_dim,
        softmax_scale,
        is_causal,
        dtype=cutlass.Float16,
        seq_len=None,
        arch=90,
    ):
        self.arch = arch
        if arch == 100:
            from .cutedsl_flash_attention_sm100 import FlashAttentionFwdSm100Aot

            self.high_batch = FlashAttentionFwdSm100Aot(
                head_dim, softmax_scale, is_causal, dtype, seq_len=seq_len
            )
        self.small = FlashAttentionFwdSm80Aot(
            head_dim,
            softmax_scale,
            is_causal,
            dtype,
            tile_m=64,
            tile_n=128,
            num_threads=128,
        )
        self.large = FlashAttentionFwdSm80Aot(
            head_dim,
            softmax_scale,
            is_causal,
            dtype,
            tile_m=128,
            tile_n=32,
            num_threads=128,
        )
        # These kernels use cp.async and mma.sync, including on newer devices.
        self.small.fa.arch = self.large.fa.arch = Arch.sm_80

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, mLSE, stream: cuda.CUstream):
        if mQ.shape[0] <= 8:
            self.small(mQ, mK, mV, mO, mLSE, stream)
        elif cutlass.const_expr(self.arch == 100):
            if mQ.shape[0] < 768:
                self.large(mQ, mK, mV, mO, mLSE, stream)
            else:
                self.high_batch(mQ, mK, mV, mO, mLSE, stream)
        else:
            self.large(mQ, mK, mV, mO, mLSE, stream)
