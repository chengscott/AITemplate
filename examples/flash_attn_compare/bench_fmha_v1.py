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
"""Benchmark the CURRENT AITemplate flash_attention op (vendored FMHA v1).

Run in the AITemplate torch env (Python 3.8, with PYTHONPATH set to the repo
`python/` dir), e.g.:

    PYTHONPATH=/local/chengscott/AITemplate_cutlass/python \
      /local/tera/anaconda3/envs/torch/bin/python \
      examples/flash_attn_compare/bench_fmha_v1.py --out /tmp/fmha_v1_results.json

Compiles the op per shape via nvcc and times it with the AITemplate runtime.
Pairs with bench_fa4.py (py3.10) and compare.py.
"""

import argparse
import json
import math
import os
import sys

import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from aitemplate.compiler import ops
from aitemplate.testing import detect_target

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import CAUSAL, flops, SHAPES, shape_key  # noqa: E402


def reference(q, k, v, causal):
    # q,k,v: (B, S, H, D). fp32 math reference.
    d = q.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.einsum("bthd,bshd->bhts", qf, kf / math.sqrt(d))
    if causal:
        s = q.shape[1]
        mask = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=q.device), 1
        )
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", attn, vf)
    return out.to(q.dtype)


def bench_one(batch, nheads, seqlen, head_dim, causal, workdir, iters=100):
    dev = "cuda"
    torch.manual_seed(0)
    # Dense per-batch tensors for the reference.
    q = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch.float16)
    ref = reference(q, k, v, causal)

    # Pack into (total, 3, nheads, head_dim) with full (unpadded) sequences.
    total = batch * seqlen
    qkv = torch.stack([q, k, v], dim=2).reshape(total, 3, nheads, head_dim).contiguous()
    cu_seqlens = torch.arange(
        0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=dev
    )

    X1 = Tensor(
        shape=[total, 3, nheads, head_dim], dtype="float16", name="qkv", is_input=True
    )
    X2 = Tensor(shape=[batch + 1], dtype="int32", name="cu_seqlens", is_input=True)
    op = ops.flash_attention(
        batch_size=batch, dropout=0.0, max_seq_len=seqlen, causal=causal
    )
    Y = op(X1, X2)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "output"

    target = detect_target()
    test_name = shape_key(batch, nheads, seqlen, head_dim, causal)
    module = compile_model(Y, target, workdir, test_name)

    inputs = {"qkv": qkv, "cu_seqlens": cu_seqlens}
    y = torch.empty([total, nheads, head_dim], dtype=torch.float16, device=dev)
    module.run_with_tensors(inputs, [y])

    y_cmp = y.reshape(batch, seqlen, nheads, head_dim)
    max_abs = (y_cmp.float() - ref.float()).abs().max().item()
    ref_scale = ref.float().abs().max().item() + 1e-6
    ok = max_abs / ref_scale < 2e-2

    time_per_iter_ms, _, _ = module.benchmark_with_tensors(inputs, [y], count=iters)

    tflops = flops(batch, nheads, seqlen, head_dim, causal) / (
        time_per_iter_ms * 1e-3
    ) / 1e12
    return {
        "ms": time_per_iter_ms,
        "tflops": tflops,
        "correct": bool(ok),
        "rel_err": max_abs / ref_scale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fmha_v1_results.json")
    ap.add_argument("--workdir", default="./tmp_fmha_v1")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    dev_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {dev_name}  sm{cc[0]}{cc[1]}  torch {torch.__version__}")

    results = {}
    for causal in CAUSAL:
        for (batch, nheads, seqlen, head_dim) in SHAPES:
            key = shape_key(batch, nheads, seqlen, head_dim, causal)
            try:
                r = bench_one(
                    batch, nheads, seqlen, head_dim, causal, args.workdir, args.iters
                )
                print(
                    f"{key:32s} {r['ms']:8.3f} ms  {r['tflops']:7.1f} TFLOP/s  "
                    f"correct={r['correct']} relerr={r['rel_err']:.2e}"
                )
            except Exception as e:  # noqa: BLE001
                r = {"error": f"{type(e).__name__}: {e}"}
                print(f"{key:32s} ERROR {r['error'][:80]}")
            results[key] = r

    meta = {"impl": "aitemplate_fmha_v1", "device": dev_name, "sm": f"{cc[0]}{cc[1]}"}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
