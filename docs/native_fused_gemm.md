# Native CUTLASS projection fusions

`AIT_FUSED_GEMM_BACKEND=auto` (the default) selects native C++ CUTLASS
implementations of `gemm_rms_relu_gemm` and `gemm_rms_swiglu` on SM90/SM100,
`gemm_rms_swiglu` on SM80, and `gemm_mxfp8_swiglu` on SM100.
This compile-time choice does not change an already compiled engine.

| Value | Behavior |
| --- | --- |
| `auto` | Native CUTLASS for the supported architectures/operators above. |
| `cutlass` | Explicitly select native CUTLASS. |
| `quack` | Select the existing CuTe DSL/Quack implementation on SM90/SM100. |

There are no other values. SM80 has no Quack implementation of this operator.
The application can disable FFN fusion with `AIT_FUSED_FFN=0`.

The default policy prioritizes batches 1024–2048 while allowing modest small-batch
tradeoffs. SM80 enables framework tuning by default. SM90/SM100 retain the
validated fixed native dispatch; their additional tuning remains explicit.
The earlier requirement for a clear win at every batch governed validation of
the native port against Quack, not every subsequent tuning choice.

The native paths preserve the operators' existing shapes, weight layouts,
workspace ABI, and numerical tolerances. They are specialized implementations,
not a replacement for general GEMM dispatch. They require the checkout's CUTLASS
headers and CUDA 13 toolchain. Native Hopper builds target SM90a for WGMMA.

The implementation includes:

- Static persistent scheduling with serpentine groups and an incomplete final
  group, avoiding the extra tiles introduced by power-of-two swizzle padding.
  Clusters contain one CTA. The small Hopper FP16 variant retains the standard
  CUTLASS scheduler required by its kernel schedule.
- Architecture- and row-count-dependent FP16 tiles and Hopper ping-pong
  scheduling for large inputs. Large FP16 variants keep residual-load and
  output-store buffers independent so stores do not delay the next residual load.
- Small SM90 producers keep operand and residual shared-memory storage separate.
  Independent producer warps issue both TMA streams concurrently, avoiding the
  residual load waiting for the GEMM operand pipeline to drain.
  The small SM90 boundary consumer uses a 64-row tile through 1296 rows,
  retaining the 128-row tile above that threshold to avoid an extra CTA wave.
- A dedicated SM100 static kernel with independent TMA-load, MMA, residual-load,
  and epilogue warp loops. It avoids the general kernel's dynamic scheduler
  pipelines and load-order barrier.
- FP16 residual, RMS reduction, ReLU, and projection epilogues using CUTLASS
  visitor trees, with padded reduction workspace for partial row tiles.
- A native MXFP8 SwiGLU epilogue that moves accumulators directly from TMEM to
  registers, applies packed arithmetic, computes block scales, and writes
  quantized values without allocating the generic shared-memory output buffers.
  Producer N tiles are 64 for up to 648 rows, 128 through 10368 rows, and 256
  above that threshold. The down projection uses N64 through 10368 rows and
  N192 above that threshold. Narrow producer tiles prefetch their row scale
  after the dependency wait, before waiting for MMA; wide tiles retain the
  later load to avoid extending register lifetimes.
- For 256-column MXFP8 tiles, full accumulator halves alternate in TMEM. The
  next tile's scale factors reuse the previous accumulator's first columns after
  the epilogue has loaded them. A final completion barrier protects deallocation
  after the early accumulator release.
- Explicit initialization of unused MXFP8 scale slots read by partial K tiles.
- Native FP16 RMS/SwiGLU, with optional residual projection and a reduction of
  the FP16-rounded residual output. Paired TMA loads retain the existing
  contiguous gate-then-up weight layout, including weights supplied as inputs.
  Hopper uses register epilogues with four-lane packing into complete
  32-byte output sectors; Blackwell uses
  direct TMEM loads, packed FP32 activation arithmetic, and 32-byte stores
  with no L1 allocation to avoid writing each L2 sector twice. Blackwell uses
  64-column tiles through 2592 rows and 128-column tiles above that threshold.
  Large Blackwell residual producers also separate residual-load and output-store
  buffers. Both forms support the operator's configurable epsilon.

The SM90/SM100 native FP16 and MXFP8 launchers enable programmatic dependent launch.
`native_gdc.h` enables CUTLASS device-side GDC instructions before its headers
are parsed and asserts that they are active on the supported architectures.
The launch attribute alone does not enable these instructions. The SM100
FP16 epilogue waits before callbacks access auxiliary global vectors. Its
full-width 128x192 producers use a row-local RMS reduction with four partial
sums, preserving FP16 residual rounding. The 64-column SwiGLU epilogue fetches
normalization before waiting for MMA. Hopper uses M64/N128 through 648 rows
and M64/N192 through 1296 rows after the smallest M64/N64 case. It uses
M128/N64 ping-pong through 1792 rows, M128/N128 through 2592 rows, then
M128/N192 for larger inputs. The narrower intermediate tile improves H200
full-model latency by roughly 0.6–5 microseconds at batches 17–22 in paired
comparisons across two checkpoints; it does not address the batch-32 gap.
For 1297–2592 rows, the Hopper ping-pong consumer loads row normalization
scales before MMA, after waiting for the preceding grid. This overlaps the
auxiliary work with the other consumer's MMA while retaining FP16 rounding.
The boundary consumer retains its existing tile order.

The three native projection operators compile and execute with Quack imports
blocked. Their native emitters bypass the CuTe/Quack exporters entirely. Other
operators can still require Quack: the current FlashAttention Python interface
imports its compilation helpers. This flag therefore does not remove Quack
from every model build.

Validation on H200 and GB200 included operator reference checks, dynamic batch
changes with reused workspace, CUDA graph replay, 4320 Hopper scheduler coverage cases (including swizzle 32),
and full-model comparisons across 17 batch sizes. Tested MXFP8 full-model outputs
matched the existing implementation exactly; FP16 differences passed the
comparison tolerances. Exact equality is not an API guarantee.

Controlled GB200 FP16 comparisons using one model's identical buffers and weights
show native ahead at all 17 tested batches, from 1 through 2048, in two runs.
The second run changes checkpoint, input distribution, graph capture order, and
allocation placement. Speedups were 0.6–2.0% in the first run and 0.5–2.0% in the
second. These measurements apply to the specialized operators and tested model,
not arbitrary GEMMs or allocations.

On H200, concurrent operand/residual loads, early normalization, and selective
single-CTA cluster launches improve complete-model graph replay. Producers and
the boundary consumer request explicit clusters for 1793–2560 rows; the SwiGLU
consumer does so for 1793–2592 rows. Beyond 2560 rows, clustering the boundary
consumer loses performance, so the launch policies intentionally differ.
All paths retain programmatic dependent launch.

The final H200 native implementation wins all 17 tested batches in two controlled
full-model sweeps. Batch 32 measures 613.16 versus 613.76 microseconds in the first
run and 611.72 versus 614.91 microseconds in the second (native versus existing).
A separate test with 16 independent captures per backend measures 612.78 versus
614.16 microseconds. A capture-level bootstrap 95% interval places the native
advantage at 0.88–1.87 microseconds in that experiment. Individual captures vary;
these results establish an average advantage for the tested workload and setup,
not a guarantee for every allocation or arbitrary GEMM.

Native MXFP8 on GB200 also wins all 17 batches in two full-model comparisons,
with exact outputs in those runs. The launch refinements above apply only to
SM90; Blackwell paths retain their validated dispatch. Native operator tests
with Quack imports blocked, threshold/partial-row tests, changing-input graphs,
and memory/race sanitizer checks validate the affected paths.

A subsequent stricter audit used eight independent captures per backend in each
of two runs at every batch. All 51 configuration/batch cases (H200 FP16, GB200
FP16 and GB200 MXFP8, 17 batches each) favor native with capture-level bootstrap
confidence intervals adjusted across the full set of comparisons. Each run also
passes its individual 95% confidence check. Numerical checks pass throughout;
MXFP8 outputs match exactly in these comparisons. This audit controls identical
model buffers for MXFP8 as well as FP16 and does not count statistical parity as
a performance win. The native implementation is now selected by `auto`; `quack` remains available
for explicit comparisons.

## Ampere FP16

The SM80 SwiGLU implementation shares the input tile between two CUTLASS
multistage GEMMs and consumes their FP32 accumulators in a single RMS/SwiGLU
epilogue. Only the final 576-column FP16 activation is stored; the intermediate
1152-column gate/up tensor is eliminated. The supported input width is 192,
with weights shaped `[1152, 192]` in gate-then-up order. Normalization gain must
already be folded into those weights. Biases are not supported.

SM80 uses `cp.async` pipelines and ordinary stream ordering. It does not use
TMA, WGMMA, or programmatic dependent launch. Its optional residual projection
uses a separate CUTLASS GEMM. The framework tuner searches CTA tile sizes,
three/four pipeline stages, and separate versus inline RMS reduction. As on the
newer architectures, only exact profiled row counts receive tuned dispatch;
other row counts use the fixed 64x64 three-stage kernel with separate RMS.
The tuner fallback is this native kernel, not the application's unfused graph.
Applications must still validate complete-model latency before enabling it.

All 16 candidate variants passed independent FP32-reference checks in plain and
residual modes on A100 PCIe, including partial tiles and graph replay. Changing
inputs, zero rows within an input, and non-default epsilon were also checked.
CUDA memory and race checks passed. Native SM80 projection code does not import
Quack. FP8/MXFP8 and the SM90/SM100 boundary fusion are not enabled on SM80.

Two complete-model A100 PCIe comparisons across 17 batches, using different
checkpoints and input distributions, show roughly 2.2–6.5% lower latency at
batches 16–2048 with framework-selected kernels. Batches 1–3 do not establish a
clear win. Policy/value comparisons pass throughout. These measurements validate
the specialized FP16 path. It is now enabled by default, accepting the small-batch
tradeoff in favor of the larger-batch gains.

## Compile-time tuning

Set `AIT_NATIVE_FUSION_TUNING=1` together with `AIT_FUSED_GEMM_BACKEND=cutlass`
when compiling to profile `gemm_rms_swiglu`, `gemm_rms_relu_gemm` (SM90/SM100),
and `gemm_mxfp8_swiglu` (SM100) through AITemplate's compiler profiling hooks.
The search covers supported GEMM tile/schedule combinations, separate Hopper
producer/consumer cluster launch choices, and MXFP8 producer/down-projection
N tiles. Candidates time the fused operator sequence, or a recognized connected
pair, in CUDA graphs.
They must pass numerical comparisons against the existing native dispatch in
both eager execution and graph replay. The original dispatch remains a candidate;
a replacement must exceed a 0.5% margin and the measured capture variation.
This measures a subgraph, not the complete application graph.

The optional `profile_rows` constructor argument specifies exact flattened row
counts (all input dimensions except the last). For example, an input shaped
`[batch, 81, 192]` uses `profile_rows=[81, 162, 243, 324]` to tune batches 1–4.
Without an explicit list, static shapes profile their sole row count; a single
dynamic dimension profiles its declared values and powers of two within its
bounds. Multiple dynamic dimensions profile the minimum and maximum products.
This policy is independent of the general GEMM `DynamicProfileStrategy`.
Generated code uses exact-row specializations, with the original dispatch for
unprofiled shapes. Passing an empty list disables tuning for that operator.

Results are stored in AITemplate's SQLite profile cache. Keys include the
operator, epsilon, residual-fusion mode, row count, GPU properties, driver/CUDA
and compiler versions, compilation options, and profiler/native/CUTLASS source
fingerprints. Moving the build directory does not invalidate the source
identity. `FORCE_PROFILE=1` forces measurement; `AIT_FORCE_PROFILER_CACHE=1`
requires an existing matching entry. A cache hit avoids benchmarking, although
compiling the profiler and querying the device may still be necessary.

`AIT_NATIVE_FUSION_TUNING` defaults to `1` on SM80 and `0` on SM90/SM100.
Explicit `0` retains fixed native dispatch, including for builds without access
to the target GPU; explicit `1` enables profiling on any supported architecture. Tuning adds no runtime
benchmarking and introduces no Quack dependency to the native projection ops.
It does not tune convolution, attention, arbitrary GEMMs, or every internal
scheduling constant. Full-model validation remains necessary because adjacent
kernels, allocation placement, and cache behavior can change the best choice.

Initial full-model checks covered 17 batch sizes each for H200 FP16, GB200 FP16,
and GB200 MXFP8. Numerical checks passed, and MXFP8 outputs matched exactly.
Performance was mixed: for example, H200 batch 64 improved about 1.5%, while
H200 batch 32 and GB200 batch 64 regressed about 0.47% and 0.31%, respectively.
These operator-level selections therefore remain opt-in rather than replacing
the previously validated defaults.

The profiler now recognizes connected residual/SwiGLU/boundary pairs from tensor
edges and profiles the pair with its actual intermediate-buffer aliases and
normalization epsilons. Each candidate changes one operator while its neighbor
uses the validated fallback. An ordering kernel separates independent timing
samples; this prevents a fictitious PDL chain across repeated samples while
preserving dependent launch inside the measured pair. Standalone operators use
the same sample separation. These ordering kernels exist only in the profiler.

SM100 boundary candidates independently search producer and consumer buffer
policies for each N tile. Selecting N192 no longer implicitly selects both buffer
policies during tuning. Unprofiled dispatch retains its previous defaults.
The graph context participates in the cache identity. This is subgraph profiling;
whole-model validation is still required before replacing a deployment default.

The corrected search retains the original schedules for the two diagnosed
regression cases: H200 FP16 at batch 32 and GB200 FP16 at batch 64. Independent
operator/reference checks and full-model numerical comparisons pass across the
17 tested batches for H200 FP16, GB200 FP16, and GB200 MXFP8. Maximum full-model
policy/value differences from the CuTe comparison are 0.00195/0.00049 on H200
and 0.00269/0.00049 on GB200 FP16; MXFP8 outputs match exactly.

Repeated full-model measurements show gains at selected shapes, rather than an
all-shape win. SM90/SM100 tuning remains opt-in while fixed native dispatch is
the default. One H200 profiling run was
discarded after hardware power braking was detected; its cache entries were
removed and measured again on a healthy device. Cache identity does not detect
every change in device operating conditions, so profiling requires a
representative, uncontended GPU.

## Dynamic GEMM buckets and full-model selection

General GEMM profiling must continue past cached workload buckets. A partial cache
hit previously returned from `gemm.profile`, skipping every later bucket. Code
generation could then fill an unprofiled large bucket with a small-bucket kernel.
The profiler now preserves the cached selection and profiles every missing bucket.
Existing libraries must be rebuilt to benefit from this fix; clearing valid cache
entries is unnecessary.

`AIT_GEMM_M_BUCKETS=N` retains its logarithmic sampling by default.
`AIT_GEMM_M_BUCKET_POLICY=large` adds sampling points at 4/8, 5/8, 6/8, and 7/8 of
the dynamic dimension's upper bound while retaining all logarithmic points. Each
point is independently profiled through the existing GEMM candidate search and
cache. This policy adds profiling and compilation cost and remains optional;
more samples do not guarantee lower complete-model latency.

Select deployment builds using complete graph measurements, with representative
weights and inputs. An operator-level win can be lost to adjacent operations or
memory placement. When large batches are the deployment priority, compare an
aggregate over those batches and separately bound small-batch regressions. Keep
the simpler policy when repeat measurements do not show a meaningful difference.
