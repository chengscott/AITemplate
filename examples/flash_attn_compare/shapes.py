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
"""Shared benchmark shapes for the FlashAttention-4 vs FMHA-v1 comparison.

Each shape is (batch, nheads, seqlen, head_dim). All sequences are full length
(no padding) so the packed varlen FMHA-v1 op and dense FA4 do identical work.
head_dim is restricted to values the current FMHA-v1 op supports ({64, 128}).
"""

# (batch, nheads, seqlen, head_dim)
SHAPES = [
    (16, 16, 512, 64),
    (16, 16, 1024, 64),
    (8, 16, 2048, 64),
    (4, 16, 4096, 64),
    (16, 16, 512, 128),
    (16, 16, 1024, 128),
    (8, 16, 2048, 128),
    (4, 16, 4096, 128),
]

CAUSAL = [False, True]

DTYPE = "float16"


def flops(batch, nheads, seqlen, head_dim, causal):
    """Forward attention FLOPs (2 matmuls, QK^T and PV), causal ~= half."""
    # 2 * (B*H*S*S*D) for QK^T + 2 * (B*H*S*S*D) for PV = 4*B*H*S*S*D
    f = 4.0 * batch * nheads * seqlen * seqlen * head_dim
    return f * (0.5 if causal else 1.0)


def shape_key(batch, nheads, seqlen, head_dim, causal):
    return f"b{batch}_h{nheads}_s{seqlen}_d{head_dim}_{'causal' if causal else 'full'}"
