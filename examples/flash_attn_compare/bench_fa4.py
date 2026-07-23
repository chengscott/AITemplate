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
"""Benchmark the latest FlashAttention-4 (CuTeDSL) forward on this GPU.

Run in the Python >= 3.10 env that has flash-attn-4 / nvidia-cutlass-dsl /
quack-kernels installed, e.g.:

    /local/chengscott/envs/fa4/bin/python examples/flash_attn_compare/bench_fa4.py \
        --out /tmp/fa4_results.json

Writes a JSON of per-shape latency (ms) + TFLOP/s and a correctness flag vs a
float32 PyTorch reference. Pair with bench_fmha_v1.py (run in the py3.8 env) and
compare.py.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import CAUSAL, DTYPE, flops, SHAPES, shape_key  # noqa: E402


def get_flash_attn_func():
    """Import the FA4 CuTeDSL forward from the installed flash-attn-4 package."""
    try:
        from flash_attn.cute import flash_attn_func

        return flash_attn_func
    except ImportError as err:
        raise SystemExit(
            "FlashAttention-4 not found. Run this in a Python >= 3.10 env with:\n"
            "    pip install flash-attn-4 torch==2.7.0\n"
            "(nvidia-cutlass-dsl / quack-kernels are pulled in automatically; "
            "torch must be the cxx11-ABI 2.7+ wheel.)\n"
            f"import error: {err}"
        )


def reference(q, k, v, causal):
    # q,k,v: (B, S, H, D) -> SDPA wants (B, H, S, D); upcast to fp32.
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    out = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, is_causal=causal
    )
    return out.transpose(1, 2).to(q.dtype)


def bench_one(flash_attn_func, batch, nheads, seqlen, head_dim, causal, iters=100):
    torch_dtype = torch.float16 if DTYPE == "float16" else torch.bfloat16
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch_dtype)
    k = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch_dtype)
    v = torch.randn(batch, seqlen, nheads, head_dim, device=dev, dtype=torch_dtype)

    # First call triggers CuTeDSL JIT compilation (cached afterwards).
    out, _ = flash_attn_func(q, k, v, causal=causal)

    # Correctness vs fp32 reference.
    ref = reference(q, k, v, causal)
    max_abs = (out.float() - ref.float()).abs().max().item()
    ref_scale = ref.float().abs().max().item() + 1e-6
    ok = max_abs / ref_scale < 2e-2

    # Warmup.
    for _ in range(5):
        flash_attn_func(q, k, v, causal=causal)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        flash_attn_func(q, k, v, causal=causal)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    tflops = flops(batch, nheads, seqlen, head_dim, causal) / (ms * 1e-3) / 1e12
    return {
        "ms": ms,
        "tflops": tflops,
        "correct": bool(ok),
        "rel_err": max_abs / ref_scale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fa4_results.json")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    flash_attn_func = get_flash_attn_func()
    dev_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {dev_name}  sm{cc[0]}{cc[1]}  torch {torch.__version__}")

    results = {}
    for causal in CAUSAL:
        for (batch, nheads, seqlen, head_dim) in SHAPES:
            key = shape_key(batch, nheads, seqlen, head_dim, causal)
            try:
                r = bench_one(
                    flash_attn_func, batch, nheads, seqlen, head_dim, causal, args.iters
                )
                print(
                    f"{key:32s} {r['ms']:8.3f} ms  {r['tflops']:7.1f} TFLOP/s  "
                    f"correct={r['correct']} relerr={r['rel_err']:.2e}"
                )
            except Exception as e:  # noqa: BLE001
                r = {"error": f"{type(e).__name__}: {e}"}
                print(f"{key:32s} ERROR {r['error'][:80]}")
            results[key] = r

    meta = {"impl": "flash_attention_4_cute", "device": dev_name, "sm": f"{cc[0]}{cc[1]}"}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
