# Follow-up GPU experiments

November 2, 2026. Apple M5 Ultra, 30 CPU cores, 64 GPU cores, 256 GiB;
MLX 0.32.4, float32, `MLX_ENABLE_TF32=0`. This extends the
[previous optimization report](NANOGPT_PERFORMANCE.md).
[MLX](https://github.com/ml-explore/mlx) is Apple's array framework;
[ddx](https://github.com/xqlsystems/ddx) still supplies SQL gradients and
[EinFold](https://github.com/xqlsystems/einfold) lowers the numerical SQL.

## Retained changes

`TrainingSession.prepare_step(capture_intermediates=False)` returns only
loss, validation status and updated state from the compiled graph. It
retains validation of every intermediate, including gradients and masks.
The existing default retains all inspection outputs for correctness tools.
Use `--discard-intermediates` in the Shakespeare benchmark. Checkpoint
resume permits changing this inspection option.

This reduces steady MLX active allocation from 532,343,668 to 314,772,687
bytes, **40.9%**, in the initial paired runs. RSS and allocator cache do
not show an equivalent reduction; this is specifically active MLX memory.
Warm time changes little from output pruning alone.

Integer joins into proven contiguous domains now compute offsets directly,
avoiding searchsorted and indirect mapping gathers. The fast path requires
int64 values and a sorted, unique, contiguous integer domain. Values are
clipped before subtracting the lower bound to avoid overflow; validity
still checks the original values. Other domains/types retain the existing
path. `XQL_MLX_DIRECT_INDEX=0` supplies the reference path for experiments.

## Complete training measurements

Each row completed 100 updates of the same 10,745,088-parameter Shakespeare
recipe, batch 12/context 256, exact GELU, dropout .2, clipped AdamW. Warm
time is the mean of updates 2–100. Runs were serial. Evaluation, batch
upload, oracle exports and checkpoint I/O are outside these timers.

| Variant, in execution order | Warm ms/update | Last active MLX MB |
|---|---:|---:|
| Fresh reference, inspection outputs retained | 108.99 | 532.34 |
| Pruned outputs | 108.20 | 314.77 |
| Direct indexing + explicit matmul experiment | 123.14 | 314.48 |
| Direct indexing, generic contractions | 102.20 | 314.48 |
| Explicit matmul only | 130.24 | 314.77 |
| Direct indexing repeat | 95.69 | 314.48 |
| Pruned-output reference repeat | 100.88 | 314.77 |
| Direct indexing + larger command buffers | 94.53 | 314.56 |

Direct indexing is **1.059× and 1.054× faster** than its corresponding
pruned-output reference. Absolute times drift between rounds; this is two
paired observations, not a confidence interval. The comparison supports
about **5–6%** additional throughput improvement, rather than a large
new multiplier. All timed steps continue to export zero numerical arrays
and read back zero array bytes, with validation and loss scalar reads.

Explicitly lowering eligible two-operand einsums into batched matmul
regressed performance and was removed. The archived experimental diff is
[available for reproduction](benchmarks/apple/next-kernel-experiments.patch).
Apply it to the preceding integration runtime with `git apply --unidiff-zero`, then use
`XQL_MLX_MATMUL=1` and optionally `XQL_MLX_DIRECT_INDEX=1`.

Larger buffers (`MLX_MAX_OPS_PER_BUFFER=1000 MLX_MAX_MB_PER_BUFFER=400`)
gave only a small additional improvement against the latest direct-index
run, while allocator cache grew to 10.41 GB and peak to 8.55 GB. They are
an explicit experiment, not a new default. The controls are implemented in
[MLX's command-buffer code](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp).

## Fusion and precision experiments

The [operator microbenchmark](benchmarks/apple/operator_candidates.py)
tests representative shapes independently of SQL lowering:

| Candidate | Compiled ordinary operations | MLX specialized kernel |
|---|---:|---:|
| Layer norm, 3,072×384 | 0.266 ms | 0.243 ms, 1.09× |
| Causal attention, 12×6×256×64, no dropout | 0.574 ms | 0.261 ms, 2.20× |

Both float32 comparisons pass 3e-5 tolerance on the sampled inputs.
These are forward microbenchmarks, not measured training improvements.
The current recipe includes probability dropout. MLX's
[fast attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
has no dropout argument. Integrating it requires a semantics-preserving
dropout/backward lowering, which is not implemented here. Layer-norm's
small standalone gain also needs full backward profiling before a new
compiler pattern is justified. No framework autodiff replaces ddx.

For a 3,072×384 by 384×1,536 product, BF16 took 0.392 ms versus float32
0.353 ms and introduced 0.288% relative L2 error. BF16 was neither faster
on this shape nor enabled in the training path. Global validation fusion
was not implemented: existing checks remain intact while the trace is
used to guide a later kernel change.

## Profiling and correctness

`--metal-capture /path/step.gputrace` captures the second training step;
launch with `MTL_CAPTURE_ENABLED=1`. A successful warmed-step capture is
at `/private/tmp/xql-next-step.gputrace` (4.3 GB, kept locally).
An Instruments Metal System Trace is at `/private/tmp/xql-next-metal.trace`.
The whole diagnostic process ran ten updates and one evaluation batch per
split. Its export contains 3,403 Python compute intervals, with 0.916 s
of interval union. This includes initialization/evaluation and does not
isolate a warm step. Exported labels do not identify individual kernel
families, so no operator-level time percentages are claimed.
See the [trace summary](benchmarks/apple/next-trace-summary.json).

The final non-integration suite passed **717 tests**, 277 optional skips,
102 integration tests deselected. A five-step inspection run checked every
loss, raw gradient, weight and optimizer moment against original nanoGPT
and PyTorch. The five-step production-output run matches those final
weights/moments within 3e-5. All 100-step experimental checkpoints also
match the fresh reference within that tolerance, including the rejected
matmul and buffer variants.

The [summary](benchmarks/apple/next-summary.json) and recorded metrics retain
timings and state-comparison errors. Recompute with
`python benchmarks/apple/summarize_next.py`; add `--check-checkpoints` when
the local, uncommitted checkpoint files are available. Run the retained path:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --discard-intermediates --steps 100 --profile-memory
```

GPU capture/trace runs are diagnostic and excluded from performance tables.
No new changes were needed in ddx or EinFold for these retained improvements.
