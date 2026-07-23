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
"""Compare the two flash_attention backends BOTH embedded in AITemplate .so files:

  * FMHA v1  — the vendored 2022 CUTLASS C++ kernel (use_cutedsl_attention=False)
  * FA4      — FlashAttention-4 SM80 forward, AOT-compiled CuTeDSL linked into the
               model .so (use_cutedsl_attention=True)

Same op (`ops.flash_attention`), same packed-QKV inputs, same AIT runtime timing
(`benchmark_with_tensors`) — an apples-to-apples in-generated-code comparison.

Requires the py3.10 build env (flash-attn-4 + nvidia-cutlass-dsl) since the FA4
backend AOT-compiles at AIT build time. Run:

    PYTHONPATH=$(pwd)/python CUDA_HOME=/usr/local/cuda \
      /local/chengscott/envs/fa4/bin/python \
      examples/flash_attn_compare/bench_ait_v1_vs_fa4.py --workdir ./tmp_cmp
"""

import argparse
import math
import os
import sys

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import CAUSAL, flops, SHAPES, shape_key  # noqa: E402


def reference(q, k, v, causal):
    d = q.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.einsum("bthd,bshd->bhts", qf, kf / math.sqrt(d))
    if causal:
        s = q.shape[1]
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshd->bthd", attn, vf).to(q.dtype)


def build_and_run(batch, nheads, seqlen, head_dim, causal, use_cutedsl, workdir, iters):
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    ref = reference(q, k, v, causal)

    total = batch * seqlen
    qkv = torch.stack([q, k, v], dim=2).reshape(total, 3, nheads, head_dim).contiguous()
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=dev)

    X1 = Tensor(shape=[total, 3, nheads, head_dim], dtype="float16", name="qkv", is_input=True)
    X2 = Tensor(shape=[batch + 1], dtype="int32", name="cu_seqlens", is_input=True)

    target = detect_target(use_cutedsl_attention=use_cutedsl)
    with target:
        op = ops.flash_attention(batch_size=batch, dropout=0.0, max_seq_len=seqlen, causal=causal)
        Y = op(X1, X2)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

    tag = "fa4" if use_cutedsl else "v1"
    name = shape_key(batch, nheads, seqlen, head_dim, causal) + "_" + tag
    with compile_model(Y, target, workdir, name) as module:
        y = torch.empty([total, nheads, head_dim], dtype=torch.float16, device=dev)
        module.run_with_tensors({"qkv": qkv, "cu_seqlens": cu}, {"output": y})
        y_cmp = y.reshape(batch, seqlen, nheads, head_dim)
        rel = (y_cmp.float() - ref.float()).abs().max().item() / (
            ref.float().abs().max().item() + 1e-6
        )
        ms, _, _ = module.benchmark_with_tensors(
            {"qkv": qkv, "cu_seqlens": cu}, {"output": y}, count=iters
        )
    tflops = flops(batch, nheads, seqlen, head_dim, causal) / (ms * 1e-3) / 1e12
    return {"ms": ms, "tflops": tflops, "correct": rel < 2e-2, "rel_err": rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="./tmp_cmp")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    dev_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {dev_name}  sm{cc[0]}{cc[1]}  torch {torch.__version__}\n")

    hdr = (
        f"{'shape':30s} {'v1 ms':>8s} {'FA4 ms':>8s} {'speedup':>8s} "
        f"{'v1 TF/s':>8s} {'FA4 TF/s':>8s}  {'ok':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for causal in CAUSAL:
        for (batch, nheads, seqlen, head_dim) in SHAPES:
            key = shape_key(batch, nheads, seqlen, head_dim, causal)
            try:
                v1 = build_and_run(batch, nheads, seqlen, head_dim, causal, False, args.workdir, args.iters)
                fa4 = build_and_run(batch, nheads, seqlen, head_dim, causal, True, args.workdir, args.iters)
                sp = v1["ms"] / fa4["ms"] if fa4["ms"] > 0 else 0.0
                ok = ("V" if v1["correct"] else "v") + ("F" if fa4["correct"] else "f")
                print(
                    f"{key:30s} {v1['ms']:8.3f} {fa4['ms']:8.3f} {sp:7.2f}x "
                    f"{v1['tflops']:8.1f} {fa4['tflops']:8.1f}  {ok:>4s}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"{key:30s} ERROR {type(e).__name__}: {str(e)[:60]}")
    print("\nspeedup = v1_ms / FA4_ms (>1 => FA4 faster). ok: V/F correct vs fp32 ref")


if __name__ == "__main__":
    main()
