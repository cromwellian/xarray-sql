# Predicate proofs and GPU gradient accumulation

This follows the [expression-sharing report](GPU_COMPILER_OPTIMIZATIONS.md).
The pipeline remains SQL → [DataFusion](https://datafusion.apache.org/), a
SQL engine → [ddx](https://github.com/xqlsystems/ddx), a SQL automatic
differentiator → [EinFold](https://github.com/xqlsystems/einfold), a compact
tensor optimizer → [MLX](https://github.com/ml-explore/mlx), Apple's GPU array
framework. SQL and derivative formulas, full validation and float32 precision
are retained. No handwritten kernel or framework autodiff replaces the SQL.

## What changed

Inspection of the actual forward/backward graph found repeated Boolean
CASE expressions and `COUNT(...) > 0` chains. These preserve SQL row and
NULL semantics, but some have simpler provably equivalent forms.

EinFold now normalizes literal negation, double negation and CASE expressions
with opposite Boolean branches. Explicit Boolean casts preserve result
types. This is enabled by default. A separate opt-in presence proof
recognizes exact counts whose mask depends only on kept axes: across a
nonempty reduced domain, their positive predicate is that mask. Empty
eligible domains give FALSE; dynamic reduced-axis masks and possible
integer overflow retain the original computation. Rename/gather operations
transform the fact with the count. No numerical values are copied to the
host to establish these facts.

The host independently uses direct addition when grouped UNION branches
have exactly matching ordered coordinates. Reordered or partial domains
retain indexed scatter-add. Row/NULL masks, duplicate branch contributions,
initial zero and branch order remain. See the separate
[identity-layout report](GPU_UNION_OPTIMIZATION.md) for that isolated change.

The selected defaults are Boolean normalization plus direct accumulation.
Count-presence proofs remain opt-in after measuring their allocation cost.
General matrix packing, contraction-tree planning and attention/normalization
primitive recognition are not implemented in this increment.

## Complete training measurements

Apple M5 Ultra, 30 CPU/64 GPU cores, MLX 0.32.4, float32,
`MLX_ENABLE_TF32=0`, November 2, 2026. This is the
**10,745,088-parameter Shakespeare character model**, not the earlier
124M-parameter synthetic workload. Batch 12, context 256, six layers/heads,
width 384, vocabulary 65, dropout .2, exact GELU, clipping and AdamW.

Each independent run starts from scratch and advances through exactly
**100 training updates** with the original 2,000-iteration learning-rate
schedule. GPU dropout, compiled owned state, frozen context and **full
validation** are enabled. Warm means use updates 2–100 and include forward,
ddx backward, optimizer and synchronization of new state/loss. Input
preparation, evaluation, logging and checkpoint I/O are excluded.

| Predicate mode | Union accumulation | Warm ms/update | First update s | MLX peak GB |
|---|---|---:|---:|---:|
| Off | Scatter reference | 87.56 | 1.33 | 4.36 |
| Boolean | Scatter | 82.69 | 1.21 | 4.36 |
| Presence | Scatter | 83.63 | 1.57 | 7.16 |
| Both | Scatter | 79.11 | 2.89 | 7.16 |
| Off | Identity dispatch | 67.52 | 3.45 | 4.36 |
| Boolean | Identity dispatch | 63.91 | 1.50 | 4.36 |
| Both | Identity dispatch | 61.88 | 7.71 | 7.16 |

A repeated reference/default pair measured **85.58→61.27 ms**,
**1.397× faster warm training**. The initial pair was 1.370×. Total time
inside the step timer for all 100 updates, including compilation, improved
9.997→7.824 seconds initially (1.278×) and 9.609→7.130 seconds in the
repeat (1.348×). Input+step speedups were 1.35–1.37×; these are not whole
application speedups including initialization and evaluation.

Cold timings are variable. The selected default's first step measured
1.50 seconds initially and 1.06 seconds in the repeat, versus reference
1.33/1.14 seconds. The larger proof set's 7.71-second first step later
repeated at .98 seconds. Therefore these observations do not establish a
reliable startup improvement or a permanent compilation penalty.

The selected mode's peak allocation remained about 4.36 GB in these runs.
The allocator cache at the end of the repeated run increased from about
5.41 to 6.08 GB; there is no broad memory-saving claim. Count-presence proofs
raised observed peak allocation to 7.16 GB while providing little additional
benefit once layout dispatch was enabled, so they are not the default.
Serial runs are observations, not confidence intervals.

All timed steps retained zero numerical-array readback bytes and zero
intermediate exports. Full validation reads one scalar status and loss
reporting reads one scalar loss per update.

## Graph evidence and rejected experiment

The reference used 1,581 UNION scatter additions over evaluation/tracing.
The identity path used 1,401 direct additions and retained 180 scatters;
156 dependent-group scatter reductions remain unchanged. Boolean
normalization recorded 550 proofs; optional presence reasoning recorded
12,227. GPU contraction dispatches remained 6,760. These are full-run
graph-construction counters, not warm-step kernel counts or operator-time
percentages.

We also tested bypassing nested pointwise compilation inside the outer
training compiler. Separate 100-update runs initially suggested a benefit,
but a controlled same-process comparison contradicted that interpretation.
It replayed identical complete-step inputs in 24 ABBA groups of 10
synchronized executions, 120 per variant: nested compilation measured
**59.54 ms**, inlining **64.97 ms**, about **9.1% slower**. Packed outputs
agreed within 1.9e-9. These are fixed-input replays, not a progressing
training trajectory; they are reported separately from the table.

Inlining was removed. The production backend retains nested pointwise
compilation; there is no active inlining switch. The unapplied
[experiment patch](benchmarks/apple/pointwise-inline-experiment.patch),
[portable replay harness](benchmarks/apple/compare_pointwise_compile.py) and
[raw paired results](benchmarks/apple/pointwise-replay.json) preserve the
experiment. Apply the patch with `git apply --unidiff-zero` before using
the harness. Its portable version was smoke-checked after the recorded run.
[MLX's compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
supports nested compilation; changing those boundaries requires measurement.

## Correctness and repository boundaries

The full xarray-sql suite passes **770 tests**, with 277 optional skips,
102 integration deselections and 83 existing warnings. EinFold's Python
suite passes **214 tests**, with two skips. New coverage includes 96 seeded
SQL presence cases, 16 CPU/GPU dynamic-mask and empty-group cases, and eight
UNION layout/null/duplicate/snapshot cases.
The rebuilt pinned wheel matches committed source and passes 323 combined
SQL/training/binding tests, with two optional skips.

Five final-default updates passed the original
[nanoGPT](https://github.com/karpathy/nanoGPT) PyTorch oracle for every loss,
raw gradient, weight and Adam moment. Every experimental 100-update state
matches the reference within 3e-5 across all 33 exported weight/moment
arrays; maximum absolute differences are below 2.4e-7. No checkpoint files
are committed.

EinFold changes implement the proofs and their seeded SQL tests. The
independent host accumulation change is reviewed separately; this follow-up
pins the predicate feature and adds GPU tests/evidence. ddx needs no changes.
Native formatting, Clippy, docs and REUSE checks pass. The pre-existing
Rust NaN-sign behavior test still fails on this platform; excluding it
leaves 72 passing tests and four ignored.

## Reproduction

Rebuild the pinned feature with
`.venv/bin/python accelerators/einfold_mlx/build.py`. Then run the selected
defaults:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --discard-intermediates \
  --validation full --freeze-context --steps 100 --log-interval 1 \
  --profile-memory --out-dir /tmp/predicate-default
```

The reference sets `EINFOLD_DENSE_PREDICATES=off XQL_MLX_DIRECT_UNION=0`
and uses a fresh output directory. Set predicates to `presence` or `all`
for opt-in proof ablations; `boolean` is the default. Set flags before
launching the process. The ordinary query and Arrow snapshot contracts
remain unchanged.

[Summary](benchmarks/apple/predicate-summary.json),
[all per-update measurements](benchmarks/apple/predicate-steps.csv),
[generator](benchmarks/apple/summarize_predicates.py),
[full correctness log](benchmarks/apple/predicates-full-correctness.log),
[independent oracle log](benchmarks/apple/predicates-final-oracle.log).
The generator reads named local run directories/logs and optionally checks
local checkpoints with `--check-checkpoints`.
