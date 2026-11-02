# Compiler optimization of SQL training graphs

[EinFold](https://github.com/xqlsystems/einfold), a tensor optimizer for SQL,
now shares pure expressions across the compact training program produced
by [ddx](https://github.com/xqlsystems/ddx), an automatic differentiator of
SQL. [DataFusion](https://datafusion.apache.org/) still plans SQL and
[MLX](https://github.com/ml-explore/mlx) still compiles and executes the
GPU graph. This adds general compiler rewrites without handwritten kernels,
model-specific shortcuts, MLX autodiff or a precision change.

## Implementation and scope

EinFold interns ordered expressions and typed scalar constants within one
invocation, collapses repeated Boolean predicates and rename chains, and
shares contractions with identical factors and row/output domains. The
graph retains reduction dependencies rather than treating every reduction
as an unrelated closure. Sharing is bounded to 32,768 entries and expires
with the invocation; numerical inputs are not cached by content. Existing
computed-array limits, SQL fallback and publication checks remain.

Optional count proofs replace integer counts over known full/empty domains
with exact bounded constants. They preserve empty-input NULL SUM/AVG,
separate row/column validity and dynamic masks. Counts alone did not improve
timing reliably, so expression sharing (`cse`) is the default, while
`facts` and `all` are opt-in. `EINFOLD_DENSE_OPTIMIZE=off` selects the reference
path. Choose the environment variable before launching Python.

xarray-sql exposes the optimizer counters in existing measurements and
pins the feature source in the EinFold builder. No MLX numerical operators,
SQL expressions or ddx formulas changed in this round. No ddx repository
change is required. These passes apply to the experimental compact executor;
they do not replace DataFusion's ordinary optimizer.

This is the first compiler increment. General layout planning, numerical
primitive recognition, backend autotuning and joint forward/backward
scheduling are not implemented here. Inspection of MLX 0.32.4's
[contraction implementation](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/einsum.cpp)
confirmed that individual einsums already use a greedy contraction planner.
That planner is retained. A broader positional contraction-sharing
experiment did not help this training workload and was removed.

## Training measurements

Apple M5 Ultra, 30 CPU/64 GPU cores, MLX 0.32.4, float32,
`MLX_ENABLE_TF32=0`, November 2, 2026. The model has **10,745,088 parameters**:
six layers, six heads, width 384, character vocabulary 65, batch 12,
context 256, exact GELU, dropout .2, clipping and AdamW. This is not the
earlier 124M-parameter synthetic benchmark. Each run starts from scratch
and executes exactly **100 training updates**, retaining the original
2,000-iteration learning-rate schedule.

Warm means use updates 2–100 and include forward, ddx backward, clipping,
AdamW and synchronization of new weights/moments/loss. Input preparation,
evaluation, logging and checkpoint export are outside the step timer.
Runs were serial; these observations are not confidence intervals.

| Recipe | Reference ms/update | Sharing ms/update | Speedup |
|---|---:|---:|---:|
| CPU-generated matched dropout masks, validation off | 104.30 | 97.75 | 1.067× |
| Same comparison, repeat | 107.63 | 96.59 | 1.114× |
| Final source after integer-check identity hardening, same repeat reference | 107.63 | 97.48 | 1.104× |
| GPU dropout, full validation | 94.56 | 89.64 | 1.055× |

CPU-mask input+step time improved 356.84→350.97 ms in the first pair and
360.99→349.74 ms in the repeat, only 1.02–1.03× because mask preparation
dominates that recipe. GPU-mask input+step time improved 97.54→92.93 ms,
1.050×. All timed steps retained zero numerical-array readback bytes and
zero intermediate exports. Scalar loss reads remain; full validation
also reads one scalar status.

Cold costs matter. CPU-mask first steps were 1.39–1.49 seconds for reference
and 1.25–1.36 seconds for sharing. The GPU-dropout/full-validation pair
regressed **1.37→2.22 seconds** on its first step. Its total 100 timed
updates therefore regressed **10.73→11.10 seconds**, despite the warm gain.
The observed warm saving amortizes that penalty after roughly 174 total
updates. First-step timing includes tracing/compilation and execution, not
initial evaluation or whole-application startup.

Tracked computed-array cache volume fell **11.11→7.53 GB**, about 32%.
This is logical tensor-volume accounting, not measured physical GPU memory.
MLX peak allocations varied substantially: reference runs alone ranged
2.70–4.43 GB. One pair showed a reduction and another an increase; there
is **no reliable physical-memory saving claim**. Peaks also include
initial evaluation/compilation.

Graph-construction counts cover the whole run, including initial evaluation
and preparation. In the final CPU-mask run, sharing recorded 191,530 hits;
pointwise dispatches fell 8,703→5,938 and input-array constructions
5,176→2,708. GPU contraction dispatches stayed at 6,760. These are not
kernel counts per warm update or proof of operator-level GPU attribution.

## Ablations and CPU profiling

Count proofs alone measured 104.17 ms versus 104.30 ms for reference.
Combined sharing/proofs measured 94.30 ms, but a 3.05-second first step
and higher allocation peak. Sharing alone was the more conservative
default. The positional contraction matcher measured 101.47 ms and
5.16 GB peak, without reducing GPU contraction dispatches. Its
[experimental patch](benchmarks/apple/compiler-canonical-experiment.patch)
is retained for reproduction, not applied. It applies to the EinFold source.
Apply it there with `git apply --unidiff-zero /path/to/the/patch`.

A separate two-update `cProfile` diagnostic with GPU dropout and one
evaluation batch took 3.98 seconds. The first ddx run occupied 1.19 seconds
cumulatively; compact lowering across 409 program executions occupied
.80 seconds. These nested totals overlap. Graph interning made 87,322
calls, .042 seconds self/.072 seconds cumulative. This suggests that
startup still has substantial planning/lowering work beyond interning.
The profile includes initial evaluation and final export; it does not
attribute warm GPU time. The readable
[profile](benchmarks/apple/compiler-prepare-profile.txt) preserves details.

## Correctness

The final default passed the full xarray-sql suite: **746 passed**, 277
optional skips, 102 integration tests deselected. EinFold's Python suite
passed **118 tests**, two skips, including 64 new cases across modes,
random seeds, NULLs, empty domains, duplicate bags, signed zero, float
SUM types, changing inputs, renamed axes and cache eviction.
The rebuilt pinned wheel exactly matches committed source and passes
203 focused SQL/training/binding tests with two optional skips.

Five final-source updates passed the independent original
[nanoGPT](https://github.com/karpathy/nanoGPT) PyTorch oracle for every loss,
raw gradient, weight and Adam moment. Checkpoint comparisons cover all 33
exported weight/moment arrays after 100 updates; maximum absolute
differences were 1.79e-7 for final sharing and 1.20e-7 for the GPU-dropout
pair, within 3e-5 tolerance. Count-proof/combined/experimental checkpoints
also pass that tolerance.

EinFold Rust formatting, Clippy and documentation checks pass. Its existing
`nan_compares_by_its_sign` test still fails on this machine, reproduced
before these Python-only changes. With that test excluded, the remaining
suite passes 72 tests with four ignored. REUSE license checks pass.

## Reproduction and evidence

The EinFold builder pins the feature commit; rebuild it with:

```sh
.venv/bin/python accelerators/einfold_mlx/build.py
```

Run the reference, then repeat with `EINFOLD_DENSE_OPTIMIZE=cse` and a
different output directory:

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_OPTIMIZE=off PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --discard-intermediates --validation none --freeze-context \
  --steps 100 --log-interval 1 --profile-memory --out-dir /tmp/compiler-reference
```

For the GPU-dropout/full-validation pair, add `--gpu-dropout` and replace
`--validation none` with `--validation full` on both commands. For the
independent oracle, retain intermediates and add
`--verify-nanogpt /path/to/nanoGPT` in an environment containing PyTorch.

[Summary](benchmarks/apple/compiler-summary.json),
[per-update data](benchmarks/apple/compiler-steps.csv),
[summary generator](benchmarks/apple/summarize_compiler.py),
[correctness log](benchmarks/apple/compiler-full-correctness.log),
[oracle log](benchmarks/apple/compiler-final-oracle.log).
The generator reads the named local run directories/logs; NPZ checkpoints
remain local. `--check-checkpoints` checks state when those files exist.
The committed CSV preserves every measured update without large model
checkpoints or duplicated raw logs.
