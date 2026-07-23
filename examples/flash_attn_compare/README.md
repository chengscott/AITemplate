# FlashAttention-4 vs current AITemplate FMHA (v1) — A100

Ports the **latest** FlashAttention (FA4, CuTeDSL) into the repo and benchmarks
its forward pass against the **current** vendored attention op (FMHA v1, the
2022 CUTLASS `fmha_fprop_kernel_1xN` kernel behind `ops.flash_attention`).

## What "latest" and "current" mean here

| | current (`ops.flash_attention`) | latest (FA4) |
|---|---|---|
| Source | `python/aitemplate/backend/cuda/attention/src/` (FMHA v1, 2022) | `flash_attn/cute/` @ flash-attn 2.8.4 (`b54df16`) |
| Impl | C++/CUDA emitted by AITemplate → nvcc | pure-Python CuTeDSL, JIT-compiled |
| SM80 kernel | `run_fmha_fp16_sm80` | `FlashAttentionForwardSm80` (`flash_fwd.py`) |
| Obtained via | already vendored in repo | `pip install flash-attn-4` (pure Python, no build) |

FA4 is used straight from the installed `flash-attn-4` package (no source is
vendored into this repo — see "Environments" below).

## Environments (two, on purpose)

FA4's CuTeDSL runtime (`nvidia-cutlass-dsl`) requires **Python ≥ 3.10**; the
AITemplate compile pipeline here runs on **Python 3.8**. So each side is
benchmarked in its own interpreter and the JSON results are merged.

- **FA4** — `/local/chengscott/envs/fa4/bin/python` (Python 3.10, torch 2.7.0+cu126
  built with the cxx11 ABI, `flash-attn-4`, `nvidia-cutlass-dsl==4.6.0.dev0`,
  `quack-kernels`). Built with `pip install flash-attn-4` (no CUDA/C++ build) +
  `pip install torch==2.7.0` (the cxx11-ABI wheel required by `torch-c-dlpack-ext`;
  torch 2.6 wheels are ABI=0 and fail with `undefined symbol ...B5cxx11Ev`).
- **FMHA v1** — the repo's `torch` env (Python 3.8) with `PYTHONPATH=<repo>/python`.

Exact versions verified working on this machine (A100, driver 575, nvcc 12.9):
`flash-attn-4==4.0.0b23`, `nvidia-cutlass-dsl==4.6.0.dev0`, `quack-kernels==0.5.3`,
`apache-tvm-ffi==0.1.13rc1`, `torch==2.7.0` (cxx11-ABI), `cuda-bindings==13.3.1`.
The two envs are fully independent, so FA4's toolchain never touches the py3.8
AITemplate build.

## Reproduce

```bash
SC=/tmp/fa_compare   # any scratch dir

# 1) latest FA4 (py3.10 env)
/local/chengscott/envs/fa4/bin/python examples/flash_attn_compare/bench_fa4.py \
    --out $SC/fa4_results.json

# 2) current FMHA v1 (py3.8 AITemplate env; compiles each shape via nvcc)
PYTHONPATH=$(pwd)/python /local/tera/anaconda3/envs/torch/bin/python \
    examples/flash_attn_compare/bench_fmha_v1.py --out $SC/fmha_v1_results.json \
    --workdir $SC/tmp_fmha_v1

# 3) merge into one table
python examples/flash_attn_compare/compare.py \
    --fa4 $SC/fa4_results.json --v1 $SC/fmha_v1_results.json
```

Shapes are `(batch, nheads, seqlen, head_dim)` in `shapes.py`, full sequences
(no padding), fp16, non-causal + causal. Both sides validate against a float32
PyTorch reference (`correct=True` on every shape below).

## Results (NVIDIA A100 80GB PCIe, sm80)

`speedup = v1_ms / FA4_ms` (>1 ⇒ FA4 faster). Both correct on all shapes.

```
shape                               v1 ms   FA4 ms  speedup  v1 TF/s FA4 TF/s
-----------------------------------------------------------------------------
b16_h16_s512_d64_full                0.24     0.11    2.12x    72.98   154.89
b16_h16_s1024_d64_full               0.98     0.40    2.47x    70.48   173.83
b8_h16_s2048_d64_full                2.38     0.82    2.89x    57.85   167.24
b4_h16_s4096_d64_full                8.34     1.61    5.20x    32.95   171.20
b16_h16_s512_d128_full               0.77     0.19    4.02x    44.45   178.65
b16_h16_s1024_d128_full              3.11     0.77    4.06x    44.24   179.48
b8_h16_s2048_d128_full               6.37     1.50    4.26x    43.13   183.67
b4_h16_s4096_d128_full              19.65     2.93    6.71x    27.98   187.74
b16_h16_s512_d64_causal              0.18     0.09    2.12x    46.51    98.43
b16_h16_s1024_d64_causal             0.59     0.26    2.24x    58.62   131.45
b8_h16_s2048_d64_causal              1.20     0.52    2.28x    57.34   130.90
b4_h16_s4096_d64_causal              3.50     0.98    3.58x    39.22   140.56
b16_h16_s512_d128_causal             0.56     0.17    3.32x    30.67   101.97
b16_h16_s1024_d128_causal            1.87     0.56    3.34x    36.79   123.03
b8_h16_s2048_d128_causal             3.51     1.05    3.35x    39.17   131.22
b4_h16_s4096_d128_causal            10.08     1.96    5.15x    27.28   140.45
```

### Takeaways

- **FA4 is 2.1×–6.7× faster** than the current FMHA v1 op across the board.
- The gap widens with **head_dim=128** and **long sequences**: v1 collapses to
  ~28–44 TFLOP/s (its `o_tmp` looping path scales poorly), while FA4 holds
  ~180 TFLOP/s dense / ~140 TFLOP/s causal.
- FA4 exploits causal masking for real work-skipping; v1's causal path barely
  differs from its dense path at short seqlen.

## FA4 embedded *inside* AITemplate's generated code

Beyond calling FA4 as an external library, this branch also wires FA4 into
AITemplate's codegen so it is **AOT-compiled and linked into the model `.so`** —
the same mechanism the repo's `gemm_rcr`/`bmm` CuTeDSL backends use. Toggle it
per-op via the target:

```python
target = detect_target(use_cutedsl_attention=True)   # FA4 CuTeDSL backend
# target = detect_target()                            # vendored FMHA v1 (default)
with target:
    Y = ops.flash_attention(batch_size=B, dropout=0.0, max_seq_len=S, causal=causal)(qkv, cu_seqlens)
```

At AIT build time the backend runs `cute.compile()` + `compiled.export_to_c()`
on FA4's `FlashAttentionForwardSm80`, producing `<func>_cutedsl.h` + `.o`
(embedded cubin) that link into the model `.so`; a thin generated C++ wrapper
slices the packed QKV into strided Q/K/V views and launches the kernel. The
runtime `.so` is self-contained (no Python).

Files added for the backend:
- `python/aitemplate/backend/cuda/attention/cutedsl_flash_attention_sm80.py` — thin
  `@cute.jit` entry wrapping FA4's SM80 forward for clean AOT export.
- `python/aitemplate/backend/cuda/attention/flash_attention_cutedsl.py` — AOT compile
  + C++ wrapper + `cuda.flash_attention.*_cutedsl` registry functions.
- dispatch in `python/aitemplate/compiler/ops/attention/flash_attention.py`
  (`use_cutedsl_attention`).

Apples-to-apples benchmark (both backends through `compile_model`, same op):

```bash
PYTHONPATH=$(pwd)/python CUDA_HOME=/usr/local/cuda \
  /local/chengscott/envs/fa4/bin/python \
  examples/flash_attn_compare/bench_ait_v1_vs_fa4.py --workdir ./tmp_cmp
```

### In-generated-code results (A100 80GB, fp16, both via `compile_model`)

`speedup = v1_ms / FA4_ms`. Every shape numerically correct vs fp32 ref (`VF`).

```
shape                             v1 ms   FA4 ms  speedup  v1 TF/s FA4 TF/s
b16_h16_s512_d64_full             0.235    0.112    2.09x     73.0    153.0
b16_h16_s1024_d64_full            0.980    0.384    2.55x     70.2    179.0
b8_h16_s2048_d64_full             2.377    0.760    3.13x     57.8    180.8
b4_h16_s4096_d64_full             8.354    1.556    5.37x     32.9    176.6
b16_h16_s512_d128_full            0.772    0.194    3.97x     44.5    176.9
b16_h16_s1024_d128_full           3.105    0.723    4.30x     44.3    190.2
b8_h16_s2048_d128_full            6.318    1.433    4.41x     43.5    191.8
b4_h16_s4096_d128_full           19.635    2.862    6.86x     28.0    192.1
b16_h16_s512_d64_causal           0.185    0.089    2.07x     46.5     96.4
b16_h16_s1024_d64_causal          0.586    0.262    2.24x     58.6    131.1
b8_h16_s2048_d64_causal           1.202    0.483    2.49x     57.2    142.2
b4_h16_s4096_d64_causal           3.503    0.944    3.71x     39.2    145.6
b16_h16_s512_d128_causal          0.560    0.166    3.38x     30.7    103.7
b16_h16_s1024_d128_causal         1.872    0.518    3.61x     36.7    132.7
b8_h16_s2048_d128_causal          3.501    0.953    3.67x     39.3    144.3
b4_h16_s4096_d128_causal         10.090    1.829    5.52x     27.2    150.3
```

**FA4-in-AITemplate is 2.07x–6.86x faster than the vendored FMHA v1**, both
compiled into a `.so` through the same op. (Raw: `results/ait_v1_vs_fa4_a100.txt`.)

### Correctness

`correctness_check.py` compiles the FA4 backend and compares its output against
(a) an fp32 einsum reference and (b) the pip `flash_attn_func` (the *identical*
FA4 kernel called directly). Across head_dim ∈ {32, 64, 128}, causal and
non-causal, and both 256-aligned and non-aligned sequence lengths, the embedded
kernel is **bit-exact** vs pip FA4 (`rel_vs_pip_FA4 = 0.0`) and ~2–6e-4 vs fp32:

```
chk_b2_h8_s512_d64_f     rel_vs_fp32=4.18e-04  rel_vs_pip_FA4=0.00e+00  OK
chk_b2_h8_s512_d64_c     rel_vs_fp32=2.66e-04  rel_vs_pip_FA4=0.00e+00  OK
chk_b3_h12_s1024_d128_f  rel_vs_fp32=4.80e-04  rel_vs_pip_FA4=0.00e+00  OK
chk_b3_h12_s1024_d128_c  rel_vs_fp32=4.43e-04  rel_vs_pip_FA4=0.00e+00  OK
chk_b2_h4_s256_d32_f     rel_vs_fp32=6.09e-04  rel_vs_pip_FA4=0.00e+00  OK
chk_b2_h8_s300_d64_f     rel_vs_fp32=5.20e-04  rel_vs_pip_FA4=0.00e+00  OK   (non-256-aligned)
```

The wrapper passes the *actual* sequence length (`max_seq_len`), not the op's
256-padded `seq_len`, so arbitrary uniform seqlens are handled (sequences must be
equal length across the batch — true varlen is not yet wired).

Scope of the embedded backend: dense / equal-length sequences (packed QKV is
read as `(batch, seq_len, 3, nheads, head_dim)`); `head_dim` and `causal` are
baked in at AOT time; requires the py3.10 build env (only at build time — the
resulting `.so` needs no Python). The build must run on the `port-cutlass-v4`
branch (CUTLASS 4.6.1) with `flash-attn-4` + `nvidia-cutlass-dsl` installed.

## Caveats

- FA4 runs its own CuTeDSL JIT (first call ~2 s, then cached); timings exclude
  compilation via warmup. v1 is timed with AITemplate's `benchmark_with_tensors`.
- Cross-interpreter torch versions differ (2.7 vs 2.3) but timing is device-side
  (CUDA events / AITemplate runtime), so the kernel comparison is fair.
- This benchmarks **forward only**, the scope the current op supports.
