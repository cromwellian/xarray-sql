# Historical Modded-NanoGPT: local throughput and numerical checks

Measured November 2, 2026, on Apple M5 Ultra, 30 CPU/64 GPU cores,
256 GiB, macOS 27.0.1. MLX 0.32.4, PyTorch 2.15.0, DataFusion 55.0.0,
NumPy 2.5.4, Python 3.12.14, `MLX_ENABLE_TF32=0`. The local EinFold build
uses commit `805f50ca6f57b9249815c3d5785e2c78724ef789`; the ddx build retains
prepared primal outputs. See the [recipe and commands](MODDED_NANOGPT.md)
for project introductions, the upstream pin and differences from the record.

## What was measured

The **123,568,128-parameter** October 14, 2024 architecture trains from
scratch on genuine GPT-2-tokenized FineWeb: 12 layers, width 768, six heads,
context **1,024**, vocabulary 50,304, **batch 1/accumulation 1**. Both paths
run all five Muon iterations, AdamW on the tied embedding, and the original
5,100-update learning-rate schedule. These are **training** measurements,
including forward, backward and optimizer, not inference.

Each run performs exactly **20 advancing updates**. The warm mean uses
updates 11–20; the first ten still train and remain in total training time.
Runs were serial in SQL → PyTorch → PyTorch → SQL order. Inputs and initial
weights match by hashes; separate runs use the same source files. Two
repeats are observations, not confidence intervals.

SQL uses a compiled GPU-resident step with **full numerical/row/NULL
validation**. PyTorch uses the actual upstream model/autograd/Muon classes,
adapted to float32 and eager MPS, with ordinary unfused AdamW. This is a
comparison with that local control, not compiled CUDA or H100 performance.

The per-update timer includes sequential input sampling, input binding/
uploads, schedule binding, forward, ddx backward, every optimizer operation,
synchronization and loss/status reads. Evaluation, checkpoint export/I/O,
oracle work and log writes are outside that timer. Setup and elapsed times
start inside `main()`, excluding interpreter startup and module imports.

## Results

| Serial pair | SQL/MLX warm ms/update | PyTorch/MPS warm ms/update | SQL/PyTorch time ratio |
|---|---:|---:|---:|
| Initial | 606.28 | 181.63 | 3.338× |
| Reversed order | 605.61 | 181.56 | 3.336× |

PyTorch/MPS is **about 3.34× faster** on this local workload. Warm throughput
is 1,689–1,691 tokens/s for SQL and 5,638–5,640 tokens/s for PyTorch.

| Initial run | SQL/MLX | PyTorch/MPS |
|---|---:|---:|
| Setup inside `main()` | 2.224 s | 1.455 s |
| First training update | 10.276 s | 0.365 s |
| All 20 timed training updates | 21.791 s | 3.839 s |
| Elapsed through final checkpoint | 27.131 s | 5.561 s |
| Peak process RSS, through checkpoint export | 24.67 GB | 2.44 GB |

The repeat observes similar totals. RSS is **process resident memory**, not
a matched physical GPU-memory peak; MLX and MPS account for device/driver
memory differently. These figures do not establish a 10× total-memory
ratio. The cold SQL trace and its metadata/storage footprint are material
remaining targets.

Every warm SQL update reports zero numerical-array readback bytes and zero
training exports inside the compiled body. One status and one loss scalar
are still read. Dynamic input binding/upload is included in the timer but
outside those body counters; zero input bytes in that counter is not a
claim that input upload disappeared. Final checkpoints are exported later.

## Correctness findings and current limit

The full existing xarray-sql suite passes **868 tests**, with 282 optional
skips, 102 network-integration deselections and 83 existing warnings.
The separate PyTorch-capable environment passes **17 focused tests**,
including five advancing updates of two-layer models on SQL CPU/GPU,
single/two-microbatch accumulation, every raw gradient, weight and moment,
changed batches, evaluation and a real-format-shard CLI/checkpoint run.
Data tests reject malformed headers, wrong sizes and out-of-vocabulary IDs.

The full-size model **does not pass the strict per-element state oracle**.
The comparison retains that failure. Across 20 updates, maximum loss
differences are 1.34e-5 and all losses pass `atol=rtol=3e-5`. However, final
147-array checkpoints fail that criterion: the initial pair has 17,196
mismatching elements, with maximum absolute difference **0.00181834**.
Matching losses alone does not establish matching training trajectories.

An independent first-update diagnostic isolates a numerical mechanism.
All 73 raw-gradient arrays pass the same 3e-5 tolerance, while nine tied
embedding weights fail after Adam. For embedding element `[1807,467]`:

| First-update quantity | SQL float32 | PyTorch/MPS float32 | Independent float64 |
|---|---:|---:|---:|
| Raw gradient | 1.456e-8 | 6.985e-10 | 1.358e-9 |

At initialization both residual projections are zero, so a separate
float64 tied-embedding → RMS norm → tied-head calculation evaluates the
same mathematical gradient at higher precision. The few near-zero differences
are consistent with float32 cancellation, and Adam's **1e-8 epsilon**
amplifies their effect on weights. This explains the inspected element;
it is not proof that every later checkpoint discrepancy has one cause.
An equivalent reciprocal-square-root SQL spelling was tried and did not
remove the discrepancy, so that experiment was not retained.

The recipe keeps its original optimizer epsilon and strict oracle tolerance;
neither is loosened to declare success. `--verify mps` on the full model
currently fails at this state check. The benchmark is useful for throughput
and exposes this correctness target, but strict full-model state parity
remains unresolved.

## Reproduction and retained evidence

Run the two commands in [the recipe](MODDED_NANOGPT.md), with
`--steps 20 --warmup 10 --checkpoint`, then run
`benchmarks/apple/compare_modded_nanogpt.py` with `--checkpoints`.
That script checks matching configurations, initial/source/shard hashes
and exact update counts before producing ratios. Add `--require-state-match`
to return a failure for the recorded checkpoint discrepancy. Fresh output
directories are required.

The [initial comparison](benchmarks/apple/modded-nanogpt-20-step-comparison.json)
and [repeat](benchmarks/apple/modded-nanogpt-20-step-repeat.json) retain full
metadata, all 80 training timing/loss samples and all checkpoint error
summaries. Large NPZ checkpoints remain local. The
[embedding diagnostic](benchmarks/apple/diagnose_modded_embedding.py) reproduces
the first-update raw-gradient/state comparison and float64 calculation:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" python benchmarks/apple/diagnose_modded_embedding.py \
  --data-dir ../modded-nanogpt/data/fineweb10B --upstream ../modded-nanogpt \
  --output /tmp/modded-gradient-diagnostic.json
```

[Diagnostic output](benchmarks/apple/modded-nanogpt-gradient-diagnostic.json)
retains per-array errors and the selected float64 references. This is outside
all performance timers.

The historic [published run log](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt)
reports validation loss 3.2741 in 910.546 timed seconds on **eight H100s**,
using 512 sequences/update and mixed precision. No ratio against that
time is reported: this local test changes hardware, precision, batch and
training budget, and performs no full validation-to-target experiment.
The new trainer supports fixed-prefix validation and effective-batch
accumulation; the full 5,100-update quality run has not been performed.

Next work is to stabilize the SQL normalization/loss derivative arithmetic,
reduce cold trace/storage costs, and profile the generated Muon GEMMs and
layouts. A full time-to-quality claim requires addressing numerical parity
and running the complete declared validation protocol.
