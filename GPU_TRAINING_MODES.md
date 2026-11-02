# Training validation, frozen execution and a matched JAX–MLX control

This predates the [graph-sharing](GPU_COMPILER_OPTIMIZATIONS.md) and
[predicate/layout](GPU_PREDICATE_OPTIMIZATIONS.md) increments. With current
code, disable those later choices using `EINFOLD_DENSE_OPTIMIZE=off`,
`EINFOLD_DENSE_PREDICATES=off` and `XQL_MLX_DIRECT_UNION=0` when reproducing
the earlier SQL dispatch configuration.

Measured November 2, 2026, on Apple M5 Ultra (30 CPU cores, 64 GPU cores,
256 GiB), macOS 27.0.1. These changes extend
[the preceding GPU experiments](NEXT_GPU_OPTIMIZATIONS.md).
[ddx](https://github.com/xqlsystems/ddx) continues to differentiate the SQL;
[EinFold](https://github.com/xqlsystems/einfold) lowers its numerical programs
to [MLX](https://github.com/ml-explore/mlx), Apple's array framework.

## Validation policies

`TrainingSession.prepare_step(validation=...)` chooses a policy before
tracing/compilation. The default remains `full`.

| Policy | Runtime validation |
|---|---|
| `full` | Check loss, named gradient/stage/update outputs and masks, as before |
| `state` | Check loss and newly published state, including their row/NULL flags |
| `none` | Trust numerical and row/NULL contracts; omit the policy's validation graph and status readback |

`none` returns no status tensor from the compiled function. It still
synchronizes device execution before publishing an update, so the timer
does not omit deferred optimizer work. Binding retains its input
shape/type/domain/finite checks. Compilation eligibility, ddx preparation
checks, and masks needed to *compute* SQL results remain. Turning off
assertions does not remove SQL's mathematical row/NULL handling.

The reduced policies deliberately weaken invalid-data rejection. An
unused named stage producing NaNs is rejected by `full`, but may be
eliminated under `state`/`none`. Invalid published state is rejected by
`full`/`state`; `none` can propagate it into later device steps. Ordinary
Arrow export retains its separate finite-output checks. Validation modes
do not change the gradient formulas or arithmetic on valid inputs.

## Frozen execution

`freeze_context=True` requires compiled training. The program assumes its
Context's schemas, coordinates, functions, views and settings remain
unchanged. Callers must prepare a new step after external Context changes;
frozen execution intentionally does not scan the catalog to detect them.
Fixed-schema numeric input replacements through `session.bind` are supported.

After the first successful step, the hot path avoids DataFusion settings
queries, catalog walks, ordinary-query setup and function-override scans.
Input layouts/slots are reused, reading current arrays rather than cached
old numerical values. Changes to the local structural generation, such as
new index-domain declarations, reject the old frozen program. Output
wrappers continue to provide snapshots; there is no in-place mutation of
previous results. Timing instrumentation remains accurate on the fast path.

Python submission, state wrappers, input binding and optional scalar loss
reads remain. This is reduced host bookkeeping, not zero host execution.

## Validation and host-overhead measurements

Each variant completed 100 updates of the real 10.7M-character model:
batch 12, context 256, 6 layers/heads, width 384, exact GELU, GPU dropout
.2, global clipping and AdamW. Inspection outputs were disabled. Warm
means use updates 2–100; batch preparation/upload, evaluation and final
checkpoint export are outside the step timer. Float32 and
`MLX_ENABLE_TF32=0` were used throughout.

| Variant | Warm ms/update | Status scalar reads/update |
|---|---:|---:|
| Full checks, ordinary context | 93.66 | 1 |
| State-only checks | 95.57 | 1 |
| No policy validation | 94.13 | 0 |
| Frozen context, full checks, cached input layouts | 89.64 | 1 |
| Frozen context, no policy validation, cached layouts | 91.61 | 0 |

Disabling validation alone produced **no reliable timing improvement**.
Frozen execution saved a few percent in these observations; removing
checks on top of it did not help in the final run. All timed steps retained
zero numerical array readback bytes. These are serial individual runs,
not confidence intervals. Earlier frozen trials, retained in the summary,
measured 91.36/90.39 ms before input-layout caching; timings vary between runs.
All variant checkpoints match the full-check reference within 3e-5.

## Matched JAX–MLX control

[JAX](https://github.com/jax-ml/jax) supplies the independent model and
automatic differentiation for the control only. The
[MetalJAX plugin](https://github.com/eterevsky/metaljax) lowers its compiled
StableHLO programs to a privately bundled, patched MLX runtime. This is a
different MLX build from SQL's public MLX 0.32.4, so the comparison does
not isolate frontend overhead alone.

An isolated Python 3.12 environment uses `metaljax==0.11.9`,
`jax==0.11.2`, `jaxlib==0.11.2`, NumPy 2.5.4,
`JAX_PLATFORMS=metal`, and `METALJAX_MATMUL_PRECISION=high`.
The harness prints and asserts `MetalDevice(id=0)` and rejects CPU fallback.
The project's existing environment and CPU JAX installation are unchanged.

Both implementations use the same SQL initializer, NumPy seeds, actual
train/validation split, training batches, exact GELU, tied embeddings,
explicit CPU-generated dropout masks, clipping, AdamW and learning-rate
schedule. The JAX initializer extracts only the trusted repository's pure
NumPy initialization functions, without importing SQL runtime packages.
Both numerically execute float32; JAX token indices are int32 and SQL
indices are int64. Iteration-zero evaluation RNG consumption is mirrored.

| Matched 100-update control | First round warm step | Repeat warm step |
|---|---:|---:|
| JAX → MetalJAX → private MLX → Metal | 40.04 ms | 40.05 ms |
| SQL → ddx/EinFold → public MLX → Metal; frozen, validation off | 101.76 ms | 94.38 ms |

SQL's step remains **2.36–2.54× slower** in these final-layout runs. A
pre-layout-cache SQL trial measured 91.78 ms. Compilation, evaluation,
input sampling/upload and checkpoint exports are outside the warm device
step timings. Both timers synchronize all new weights/moments and the loss.
The CPU dropout masks are supplied as inputs in this control; GPU-mask
recipe measurements in the preceding section are a different experiment.

The repeat also records input and complete iteration costs. Mean input
plus step time is **290.95 ms for JAX and 316.19 ms for SQL**. The much
smaller practical gap reflects expensive CPU mask generation/upload in
both controls. These figures exclude evaluation/logging/checkpoint I/O.
SQL binds float64 host mask Datasets into float32 device arrays; JAX uploads
float32 masks. This is not a whole-application throughput comparison with
native GPU-dropout nanoGPT.

Five SQL updates passed the original nanoGPT/PyTorch oracle for every
loss, raw gradient, weight and moment. The matched JAX five-step final
weights/moments agree with that SQL run (maximum absolute difference
2.82e-7). Both 100-step JAX checkpoints agree with the corresponding SQL
checkpoints at 3e-5 tolerance, maximum absolute differences below 3.6e-7.

The remaining step gap persists after runtime policy validation and
catalog scans are removed. This points toward numerical graph lowering,
remaining SQL-semantic computations and backend kernel choices, rather
than those checks being the main explanation. The private MLX build is
also a material difference. Operator-level attribution needs further
profiling; these measurements do not assign percentages to individual causes.

## Correctness and reproduction

The final suite passed **746 tests**, 277 optional skips, 102 integration
tests deselected. Coverage includes all validation/freeze/inspection
combinations with changing tokens and learning rates, finite-state failure
atomicity, missing rows, NULL outputs, explicit trusted-mode behavior,
structural-generation rejection, snapshots and checkpoint resume.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --discard-intermediates \
  --validation full --freeze-context --steps 100

# Trusted execution: policy validation and its status readback are omitted.
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --discard-intermediates \
  --validation none --freeze-context --steps 100
```

The matched comparison omits `--gpu-dropout` on the SQL command to use the
same CPU-mask stream. Its independent JAX command is:

```sh
uv venv --python 3.12 /private/tmp/xql-jax-mlx-venv
uv pip install --python /private/tmp/xql-jax-mlx-venv/bin/python \
  metaljax==0.11.9 jax==0.11.2 jaxlib==0.11.2 numpy==2.5.4

JAX_PLATFORMS=metal METALJAX_MATMUL_PRECISION=high MLX_ENABLE_TF32=0 \
  /private/tmp/xql-jax-mlx-venv/bin/python benchmarks/apple/compare_shakespeare_jax_mlx.py \
  --steps 100 --output benchmarks/apple/jax-control.json \
  --checkpoint /private/tmp/jax-control.npz
```

[Raw summary](benchmarks/apple/training-modes-summary.json),
[JAX control harness](benchmarks/apple/compare_shakespeare_jax_mlx.py),
[full correctness log](benchmarks/apple/modes-full-correctness.log).
Rebuild the summary with `python benchmarks/apple/summarize_training_modes.py`;
add `--check-checkpoints` when the local, uncommitted NPZ files are available.
