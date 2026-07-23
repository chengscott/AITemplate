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
Thin CuTeDSL entry that wraps FlashAttention-4's SM90 forward for AOT export.

This is the Hopper (SM90) counterpart of ``cutedsl_flash_attention_sm80.py``.
FlashAttention-4's ``FlashAttentionForwardSm90`` is the warp-specialized
(TMA + WGMMA) forward.  Like the SM80 wrapper, its ``__call__`` takes ~17
arguments -- several of them Python objects (``AuxData``) or ``None`` optionals
that the CuTeDSL ``export_to_c`` C-header generator cannot represent as C
function parameters.  This wrapper exposes only the C-friendly tensors + stream
and bakes everything else (softmax scale, causal flag, all optional tensors,
``AuxData``) in as compile-time constants, so ``cute.compile(...)`` +
``compiled.export_to_c`` produce a clean C interface:

    cute_dsl_<name>_wrapper(module, mQ, mK, mV, mO, mLSE, stream)

To make the embedded kernel *bit-identical* to the pip ``flash_attn_func`` (the
correctness check compares against it), the SM90 tile / stage / warp config is
taken from FlashAttention-4's own dispatcher rather than hand-picked: tile sizes
and the ``mma_pv_is_rs`` / ``intra_wg_overlap`` flags come from
``flash_attn.cute.interface._tile_size_fwd_sm90`` and the remaining knobs
(``num_stages=2``, ``num_threads=384`` = 1 producer + 2 MMA warpgroups) mirror
the ``arch // 10 == 9`` branch of ``interface._flash_attn_fwd``.

Requires the FlashAttention-4 CuTeDSL package (``pip install flash-attn-4``) and
``nvidia-cutlass-dsl`` in the build environment (Python >= 3.10).  See
``examples/flash_attn_compare/README.md``.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.interface import _tile_size_fwd_sm90
from flash_attn.cute.utils import AuxData

# Hopper warp-specialized forward: 1 producer + 2 MMA warpgroups = 384 threads,
# 2-stage TMA pipeline. Matches interface._flash_attn_fwd's ``arch // 10 == 9``
# branch (num_threads default 384, num_stages=2).
_SM90_NUM_THREADS = 384
_SM90_NUM_STAGES = 2


class FlashAttentionFwdSm90Aot:
    """AOT-exportable SM90 (Hopper) FlashAttention-4 forward.

    All non-tensor knobs are compile-time constants so the exported C function
    is ``(mQ, mK, mV, mO, mLSE, stream)``.  q/k/v/o are (B, S, H, D); lse is
    (B, H, S).  batch/seqlen/nheads stay dynamic at runtime; head_dim is static.

    The tile config (tile_m/tile_n + ``mma_pv_is_rs``/``intra_wg_overlap``) is
    derived from FA4's ``_tile_size_fwd_sm90`` for the given (head_dim, causal),
    so the kernel matches what ``flash_attn_func`` launches on Hopper.
    """

    def __init__(
        self,
        head_dim: int,
        softmax_scale: float,
        is_causal: bool,
        dtype=cutlass.Float16,
    ):
        self.softmax_scale = softmax_scale
        # Dense (no local/window) forward; head_dim_v == head_dim.
        fwd_cfg = _tile_size_fwd_sm90(
            head_dim, head_dim, is_causal, False  # is_local=False
        )
        self.fa = FlashAttentionForwardSm90(
            dtype,
            head_dim,
            head_dim,  # head_dim_v == head_dim
            1,  # qhead_per_kvhead (no GQA)
            is_causal=is_causal,
            is_local=False,
            pack_gqa=False,
            tile_m=fwd_cfg.m_block_size,
            tile_n=fwd_cfg.n_block_size,
            num_stages=_SM90_NUM_STAGES,
            num_threads=_SM90_NUM_THREADS,
            Q_in_regs=False,
            intra_wg_overlap=fwd_cfg.intra_wg_overlap,
            mma_pv_is_rs=fwd_cfg.mma_pv_is_rs,
            score_mod=None,
            mask_mod=None,
            has_aux_tensors=False,
            q_subtile_factor=1,  # no block sparsity
            paged_kv_non_tma=False,  # dense, non-paged
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
