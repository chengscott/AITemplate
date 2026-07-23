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
"""Combine bench_fa4.py and bench_fmha_v1.py JSON outputs into one table.

    python examples/flash_attn_compare/compare.py \
        --fa4 /tmp/fa4_results.json --v1 /tmp/fmha_v1_results.json
"""

import argparse
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(r, field):
    if r is None or "error" in r:
        return "  --   "
    return f"{r[field]:7.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa4", default="/tmp/fa4_results.json")
    ap.add_argument("--v1", default="/tmp/fmha_v1_results.json")
    args = ap.parse_args()

    fa4 = load(args.fa4)
    v1 = load(args.v1)
    fa4_r, v1_r = fa4["results"], v1["results"]

    print(f"FA4 : {fa4['meta']}")
    print(f"v1  : {v1['meta']}")
    print()
    hdr = (
        f"{'shape':32s} {'v1 ms':>8s} {'FA4 ms':>8s} {'speedup':>8s} "
        f"{'v1 TF/s':>8s} {'FA4 TF/s':>8s}  {'ok':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))

    keys = list(v1_r.keys())
    for k in fa4_r:
        if k not in keys:
            keys.append(k)

    for k in keys:
        a = fa4_r.get(k)
        b = v1_r.get(k)
        speedup = "  --   "
        if a and b and "error" not in a and "error" not in b and a["ms"] > 0:
            speedup = f"{b['ms'] / a['ms']:6.2f}x"
        ok = ""
        if a and "correct" in a:
            ok += "F" if a["correct"] else "f"
        if b and "correct" in b:
            ok += "V" if b["correct"] else "v"
        print(
            f"{k:32s} {fmt(b, 'ms'):>8s} {fmt(a, 'ms'):>8s} {speedup:>8s} "
            f"{fmt(b, 'tflops'):>8s} {fmt(a, 'tflops'):>8s}  {ok:>6s}"
        )

    print()
    print("speedup = v1_ms / FA4_ms  (>1 means FA4 faster)")
    print("ok column: F/V = FA4/v1 correct vs fp32 ref; lowercase = mismatch")


if __name__ == "__main__":
    main()
