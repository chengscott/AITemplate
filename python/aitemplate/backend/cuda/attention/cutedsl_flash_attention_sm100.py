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
Thin CuTeDSL entry that wraps FlashAttention-4's SM100 forward for AOT export.

Blackwell (SM100) counterpart of ``cutedsl_flash_attention_sm90.py``.
FlashAttention-4's ``FlashAttentionForwardSm100`` is the tcgen05 (UMMA) forward.
Its ``__call__`` takes ~25 arguments -- several Python objects (``AuxData``) or
``None`` optionals the CuTeDSL ``export_to_c`` generator cannot represent as C
parameters.  This wrapper exposes only the C-friendly tensors + stream and bakes
everything else in as compile-time constants, so ``cute.compile(...)`` +
``compiled.export_to_c`` produce a clean C interface:

    cute_dsl_<name>_wrapper(module, mQ, mK, mV, mO, mLSE, stream)

Unlike SM90, the SM100 forward has no ``dtype`` ctor arg (it infers dtype from
the tensors) and detects the arch internally via the DSL.  This is the dense
(equal-length, non-paged) forward, so the kernel config mirrors what
FlashAttention-4's own ``interface._get_fwd_config`` / ``_flash_attn_fwd`` pick
for a dense SM100 problem, tuned to the ``seq_len`` AIT bakes in at codegen:

  * ``tile_m``/``tile_n`` = the SM100 base tiling ``FwdConfig(128, 128, ...)``
    (arch 10.x default; head-dim independent);
  * ``q_stage`` = ``2 if seq_len > tile_m else 1`` -- FA4's SM100 rule;
  * ``use_2cta_instrs`` = ``seq_len > 2*tile_m`` -- FA4's SM100 rule (2-CTA
    cluster MMA pays off only for long sequences);
  * ``use_clc_scheduler`` = False -- conservative default;
  * ``is_static_persistent`` = ``not is_causal`` -- FA4 uses the static-persistent
    scheduler for the dense non-causal case (no varlen / no split-KV here).

When ``seq_len`` is unknown (not passed by the op), ``q_stage``/``use_2cta_instrs``
fall back to their seqlen-agnostic, correctness-safe values (1 / False).
Requires the FlashAttention-4 CuTeDSL package (``pip install flash-attn-4``) and
``nvidia-cutlass-dsl`` in the build environment (Python >= 3.10).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

# SM100 base tiling: interface._get_fwd_config's default FwdConfig(128, 128) for
# arch 10.x (head-dim independent). q_stage / use_2cta_instrs are derived from seq_len.
_SM100_TILE_M = 128
_SM100_TILE_N = 128


class FlashAttentionFwdSm100Aot:
    """AOT-exportable SM100 (Blackwell) FlashAttention-4 forward.

    All non-tensor knobs are compile-time constants so the exported C function
    is ``(mQ, mK, mV, mO, mLSE, stream)``.  q/k/v/o are (B, S, H, D); lse is
    (B, H, S).  batch/seqlen/nheads stay dynamic at runtime; head_dim is static.

    ``dtype`` is accepted for signature parity with the SM80/SM90 wrappers but is
    ignored -- FlashAttentionForwardSm100 infers the element type from tensors.
    """

    def __init__(
        self,
        head_dim: int,
        softmax_scale: float,
        is_causal: bool,
        dtype=cutlass.Float16,
        seq_len: int = None,
    ):
        self.softmax_scale = softmax_scale
        tile_m, tile_n = _SM100_TILE_M, _SM100_TILE_N
        # FA4's SM100 dispatcher (interface._flash_attn_fwd) derives these from the
        # query sequence length; fall back to the seqlen-agnostic safe values (single
        # Q stage, single-CTA MMA) when seq_len is not known at codegen.
        q_stage = 2 if (seq_len is not None and seq_len > tile_m) else 1
        # FA4's SM100 2-CTA (2-SM) MMA path is only valid when the padded head_dim is 128
        # or 192 (interface._flash_attn_fwd gates on this in addition to seqlen > 2*tile_m).
        # Guarding on head_dim too -- a seqlen-only rule would emit an unsupported 2-CTA
        # config for a head_dim outside {128, 192} (e.g. head_dim=16).
        head_dim_padded = (head_dim + 15) // 16 * 16
        use_2cta_instrs = bool(
            seq_len is not None
            and seq_len > 2 * tile_m
            and head_dim_padded in (128, 192)
        )
        # FA4 uses the static-persistent scheduler for the dense non-causal forward.
        is_static_persistent = not is_causal
        self.fa = FlashAttentionForwardSm100(
            head_dim,
            head_dim,  # head_dim_v == head_dim
            qhead_per_kvhead=1,  # no GQA
            is_causal=is_causal,
            is_local=False,
            is_split_kv=False,
            pack_gqa=False,
            m_block_size=tile_m,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_static_persistent=is_static_persistent,
            score_mod=None,
            mask_mod=None,
            has_aux_tensors=False,
            paged_kv_non_tma=False,  # dense, non-paged
            is_varlen_q=False,
            q_subtile_factor=1,  # no block sparsity
            kv_subtile_factor=1,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=False,
            has_tile_count_semaphore=False,
            seqlen_k_per_split=None,
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
        # Pass stream as a KEYWORD: the SM100 __call__ has many params between
        # softmax_scale and stream (cu_seqlens, page table, aux, semaphores, ...),
        # so a positional stream would land on the wrong slot. Optionals default
        # to None / AuxData().
        self.fa(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            self.softmax_scale,
            stream=stream,
        )
