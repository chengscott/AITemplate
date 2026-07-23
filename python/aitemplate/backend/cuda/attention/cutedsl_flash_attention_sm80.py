#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Thin CuTeDSL entry that wraps FlashAttention-4's SM80 forward for AOT export.

FlashAttention-4's ``FlashAttentionForwardSm80.__call__`` takes ~17 arguments,
several of them Python objects (``AuxData``) or ``None`` optionals that the
CuTeDSL ``export_to_c`` C-header generator cannot represent as C function
parameters.  This wrapper exposes only the C-friendly tensors + stream and bakes
everything else (softmax scale, causal flag, all optional tensors, ``AuxData``)
in as compile-time constants, so ``cute.compile(...)`` + ``compiled.export_to_c``
produce a clean C interface:

    cute_dsl_<name>_wrapper(module, mQ, mK, mV, mO, mLSE, stream)

Requires the FlashAttention-4 CuTeDSL package (``pip install flash-attn-4``) and
``nvidia-cutlass-dsl`` in the build environment (Python >= 3.10).  See
``examples/flash_attn_compare/README.md``.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.utils import AuxData


class FlashAttentionFwdSm80Aot:
    """AOT-exportable SM80 FlashAttention-4 forward.

    All non-tensor knobs are compile-time constants so the exported C function
    is ``(mQ, mK, mV, mO, mLSE, stream)``.  q/k/v/o are (B, S, H, D); lse is
    (B, H, S).  batch/seqlen/nheads stay dynamic at runtime; head_dim is static.
    """

    def __init__(
        self,
        head_dim: int,
        softmax_scale: float,
        is_causal: bool,
        dtype=cutlass.Float16,
        tile_m: int = 128,
        tile_n: int = 64,
        num_threads: int = 128,
    ):
        self.softmax_scale = softmax_scale
        self.fa = FlashAttentionForwardSm80(
            dtype,
            head_dim,
            head_dim,  # head_dim_v == head_dim
            1,  # qhead_per_kvhead (no GQA)
            is_causal=is_causal,
            is_local=False,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=1,
            num_threads=num_threads,
            Q_in_regs=False,
            score_mod=None,
            mask_mod=None,
            has_aux_tensors=False,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.fa(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            self.softmax_scale,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # page_table
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # blocksparse_tensors
            AuxData(),  # aux_data (compile-time constant)
            stream,
        )
