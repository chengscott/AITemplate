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
"""Rigorous correctness check for the FA4-in-AIT backend.

For each shape, compile the flash_attention op with the FA4 CuTeDSL backend and
compare the output against:
  (a) an fp32 einsum reference,
  (b) the pip flash_attn_func (the *identical* FA4 kernel, called directly).
Also probes a non-256-aligned seqlen to check the padded-seq_len path.
"""
import math
import sys

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from flash_attn.cute import flash_attn_func


def ref_fp32(q, k, v, causal):
    d = q.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.einsum("bthd,bshd->bhts", qf, kf / math.sqrt(d))
    if causal:
        s = q.shape[1]
        m = torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(m, float("-inf"))
    return torch.einsum("bhts,bshd->bthd", torch.softmax(scores, -1), vf).to(q.dtype)


def run(batch, nheads, seqlen, head_dim, causal, workdir):
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)

    ref = ref_fp32(q, k, v, causal)
    fa_direct, _ = flash_attn_func(q, k, v, causal=causal)  # identical kernel, direct

    total = batch * seqlen
    qkv = torch.stack([q, k, v], dim=2).reshape(total, 3, nheads, head_dim).contiguous()
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=dev)
    X1 = Tensor(shape=[total, 3, nheads, head_dim], dtype="float16", name="qkv", is_input=True)
    X2 = Tensor(shape=[batch + 1], dtype="int32", name="cu_seqlens", is_input=True)

    target = detect_target(use_cutedsl_attention=True)
    with target:
        op = ops.flash_attention(batch_size=batch, dropout=0.0, max_seq_len=seqlen, causal=causal)
        Y = op(X1, X2)
        padded = op._attrs["seq_len"]  # computed during __call__
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

    name = f"chk_b{batch}_h{nheads}_s{seqlen}_d{head_dim}_{'c' if causal else 'f'}"
    with compile_model(Y, target, workdir, name) as module:
        y = torch.empty([total, nheads, head_dim], dtype=torch.float16, device=dev)
        module.run_with_tensors({"qkv": qkv, "cu_seqlens": cu}, {"output": y})
        y = y.reshape(batch, seqlen, nheads, head_dim)

    def rel(a, b):
        return (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-6)

    r_ref = rel(y, ref)
    r_direct = rel(y, fa_direct)
    aligned = padded == seqlen
    print(
        f"{name:28s} seq_len(padded)={padded:<5d} aligned={aligned!s:<5} "
        f"rel_vs_fp32={r_ref:.2e}  rel_vs_pip_FA4={r_direct:.2e}  "
        f"{'OK' if (r_ref < 2e-2 and r_direct < 5e-3) else 'MISMATCH'}"
    )


if __name__ == "__main__":
    wd = sys.argv[1] if len(sys.argv) > 1 else "./tmp_chk"
    # aligned seqlens (padded == actual)
    run(2, 8, 512, 64, False, wd)
    run(2, 8, 512, 64, True, wd)
    run(3, 12, 1024, 128, False, wd)
    run(3, 12, 1024, 128, True, wd)
    run(2, 4, 256, 32, False, wd)
    # NON-aligned seqlen: op pads seq_len to a multiple of 256 -> probe the wrapper
    run(2, 8, 300, 64, False, wd)
