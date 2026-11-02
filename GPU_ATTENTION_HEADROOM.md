# GPU optimization status and attention headroom

November 2, 2026. Apple M5 Ultra, 30 CPU/64 GPU cores, 256 GiB,
macOS 27.0.1, MLX 0.32.4, float32, `MLX_ENABLE_TF32=0`.
[MLX](https://github.com/ml-explore/mlx) is Apple's array framework.
[ddx](https://github.com/xqlsystems/ddx) generates SQL derivatives;
[EinFold](https://github.com/xqlsystems/einfold) lowers numerical SQL planned
by [DataFusion](https://datafusion.apache.org/) into compact tensor programs.

## Implementation status

The resident training program and several compiler passes are implemented.
The broader compiler plan is **not complete**.

| Area | Implemented | Remaining work |
|---|---|---|
| Planning and CPU work | Prepared SQL/ddx plans, shared CTE execution, registration improvements and configurable thread counts | Further cold-start planning/registration work; outside warm GPU-step timing |
| GPU state | Owned weights/Adam moments, compiled whole-step execution, reused primal loss, GPU dropout | Input preparation/binding, Python submission and loss/status scalar reads remain |
| Graph optimization | Expression/contraction sharing, Boolean normalization, optional count proofs, direct accumulation for identical coordinate layouts | Numerical primitive recognition and cost-based backend selection |
| Storage and contractions | Compact dense arrays, contiguous-domain index fast path | Consumer-aware matrix layout/packing and planning across operations |
| Backward program | ddx semantics, shared subexpressions and cheaper compatible gradient accumulation | Joint forward/backward scheduling, liveness/recomputation choices and specialized derivative-block lowering |
| Checks and host overhead | Selectable validation compiled into the program, frozen context, optional inspection outputs | Further host batching; removing checks alone did not improve timing reliably |

The [training report](NANOGPT_PERFORMANCE.md),
[resident-step follow-up](NEXT_GPU_OPTIMIZATIONS.md),
[execution modes](GPU_TRAINING_MODES.md),
[graph-sharing report](GPU_COMPILER_OPTIMIZATIONS.md) and
[predicate/layout report](GPU_PREDICATE_OPTIMIZATIONS.md) record individual
measurements and limits. The latest retained compiler defaults improved
the dropout-.2 Shakespeare warm step from 85.58 to 61.27 ms in a repeated
comparison, 1.397×. These preceding production measurements are separate
from the no-dropout experiments below.

Unconditional einsum-to-matmul lowering, BF16, broad positional contraction
sharing and nested pointwise inlining were tried and rejected after
regressions or insufficient benefit. More CPU threads are configurable;
increasing them is not an established warm GPU-step improvement.

## Built-in attention eligibility

MLX exposes `mx.fast.scaled_dot_product_attention`. Its
[public API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
supports causal masks and forced fused execution, but **has no attention
probability dropout argument**. Float32 forward and all Q/K/V derivatives
work on this GPU for the measured shapes. The native MLX backward path is
selected with `MLX_SDPA_VJP_FALLBACK` unset/zero. Its
[Metal implementation](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp)
uses a blocked backward calculation with matrix multiplications; this is
not a claim that the entire backward is one fused kernel.

The older Arrow-based native attention bridge in `xarray_sql/mlx.py` already
calls this API for eligible forward operations. The resident compact
training executor still uses generic contractions and reductions for its
attention and ddx backward graph. These experiments do **not** add a
production attention rewrite or replace ddx with framework autodiff.

Probability dropout acts after softmax and before multiplying values.
A logits mask renormalizes probabilities and changes the operation.
Dropping the final attention output also changes randomness and derivatives.
Neither preserves the existing semantics. Embedding/residual dropout could
remain outside an eligible fused attention block, but the model benchmark
removes all dropout for a clear control.

## Attention-only forward and backward

The [kernel harness](benchmarks/apple/compare_attention_kernels.py) compares
resident SQL/ddx attention, native MLX ordinary operations and built-in
MLX attention. SQL computes causal scores, a stopped maximum, exponentials,
normalization and weighted values. The generic native control uses the
same stopped softmax shift. MLX autodiff is an independent control only.

Inputs are fixed seeded float32 Q/K/V and output cotangent arrays. The
forward phase returns a scalar dot loss; the backward phase returns that
loss and **all Q/K/V gradients**. There is no optimizer or full model.
Each variant receives 30 untimed warmups, then 16 groups of ten synchronized
executions. Variant order reverses each group. SQL policy validation is
omitted in this numerical comparison. No numerical arrays are exported in
the timed loop.

| B,H,T,D | Phase | SQL/ddx ms | MLX ordinary ms | MLX built-in ms | SQL / built-in |
|---|---|---:|---:|---:|---:|
| 12,6,256,64 | Forward | 0.582 | 0.423 | 0.289 | 2.02× |
| 12,6,256,64 | Forward + Q/K/V gradients | 1.377 | 0.710 | 0.549 | 2.51× |
| 12,12,256,64 | Forward | 0.769 | 0.552 | 0.303 | 2.54× |
| 12,12,256,64 | Forward + Q/K/V gradients | 2.427 | 1.088 | 0.960 | 2.53× |
| 1,12,1024,64 | Forward | 0.930 | 0.694 | 0.346 | 2.69× |
| 1,12,1024,64 | Forward + Q/K/V gradients | 3.274 | 1.415 | 1.030 | 3.18× |

Full forward outputs, scalar losses and every Q/K/V gradient match SQL
within `atol=rtol=3e-5`. The largest built-in gradient absolute error is
2.73e-6. Larger scalar dot-loss absolute errors pass the relative tolerance.
The wider/longer shapes are attention shapes only, **not complete
124M-parameter model measurements**. [Raw results](benchmarks/apple/attention-kernel-headroom.json)
include per-group timings and errors separately for each output.

## Complete no-dropout training control

The [model harness](benchmarks/apple/compare_shakespeare_mlx_attention.py)
trains the **10,745,088-parameter** Shakespeare character model from scratch:
six layers/heads, width 384, batch 12, context 256, vocabulary 65, exact
GELU, tied embeddings, clipped AdamW and the original 2,000-iteration
learning-rate schedule. Each run performs exactly 100 updates without
dropout. The SQL variant omits mask tables/operators in its benchmark-only
generator; merely supplying `--dropout 0` to the ordinary recipe retains
those operations. Production dropout behavior is unchanged.

All controls use identical initial arrays and training batches, including
matching evaluation batch RNG consumption. Native MLX does not execute
evaluation: it is a training-step control. Timers include forward/backward,
clipping, optimizer, numerical-finiteness checks, synchronized new state
and loss/status scalar reads. SQL additionally retains full SQL row/NULL
validation and uses ddx derivatives. Native controls use MLX autodiff and
do not reproduce SQL row/NULL checks. Input preparation, evaluation,
logging and checkpoint I/O are outside step timing. Warm means use updates
2–100, excluding first-step compilation.

| Variant | Warm ms/update | Input + step ms/update |
|---|---:|---:|
| Resident SQL/ddx, generic attention | 48.17 | 51.37 |
| Native MLX, ordinary attention | 20.11 | 20.25 |
| Native MLX, built-in attention | 19.12 | 19.23 |

The native manual-to-built-in improvement is **1.052×**. A second control
advances both native models through the same batches with separate state
and alternating order: **19.85 → 18.90 ms**, **1.051×**. These observations
are not confidence intervals. SQL is a separate serial run and its
comparison does not isolate attention or frontend overhead.

Every final weight and Adam moment across all four native trajectories
passes comparison with SQL: 33 arrays per trajectory, `atol=rtol=3e-5`,
maximum absolute difference below 3.6e-7. Maximum loss difference across
100 updates is 7.16e-7. SQL records zero numerical array readback bytes or
exports per training step, plus one status and one loss scalar read.
Checkpoints stay local. The [summary](benchmarks/apple/attention-summary.json)
records timing, loss and per-state errors, with hashes of raw text evidence.

The built-in primitive helps the isolated attention block. Its roughly 5%
gain in native whole-model training does not explain the remaining roughly
2.5× SQL gap. Neither result is a measured speedup of an integrated SQL
attention rewrite. Removing dropout also changes the recipe; 48.17 ms is
not a same-recipe improvement over preceding dropout-.2 measurements.

## Next implementation targets

1. Add a typed attention pattern for dense, nonnullable, compatible
   masks/scales with attention-probability dropout disabled. Start with
   evaluation/forward lowering, preserving the ordinary SQL fallback.
2. Recognize the corresponding **ddx-generated** derivative blocks and
   select equivalent backend attention backward execution. The public MLX
   API exposes this through autodiff rather than a standalone backward
   function. Integration must preserve ddx's derivative contract, SQL
   domains/masks and stop-gradient boundaries. Forward-only replacement
   would leave the existing backward cost.
3. Recognize normalization and softmax/cross-entropy blocks generally.
   Profile forward/backward together before choosing patterns.
4. Add consumer-aware layout/packing and joint backward scheduling. Packed
   Q/K/V projections are candidates; packing costs and optimizer storage
   must enter the decision.

No handwritten Metal kernel is needed for initial eligible attention.
Dropout-enabled attention retains the generic path until a backend supports
the required probability dropout and matching backward operation.

## Reproduction

From the repository root, with the pinned accelerator installed:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python \
  benchmarks/apple/compare_attention_kernels.py --groups 16 --repetitions 10 \
  --warmup 30 --output benchmarks/apple/attention-kernel-headroom.json

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python \
  benchmarks/apple/compare_shakespeare_mlx_attention.py --backend sql \
  --steps 100 --out-dir benchmarks/apple/attention-model-sql \
  --output benchmarks/apple/attention-model-sql.json

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python \
  benchmarks/apple/compare_shakespeare_mlx_attention.py --backend mlx_pair \
  --steps 100 --out-dir benchmarks/apple/attention-model-mlx-pair \
  --output benchmarks/apple/attention-model-mlx-pair.json
```

For separate serial controls, replace `mlx_pair` with `mlx_manual` and
`mlx_fast`, using corresponding `attention-model-mlx-manual` and
`attention-model-mlx-fast` output names. Defaults use prepared data at
`~/nanoGPT/data/shakespeare_char`; override with `--data-dir`. The original
[nanoGPT](https://github.com/karpathy/nanoGPT) training repository provides
the preparation recipe. Clear `MLX_SDPA_VJP_FALLBACK` when measuring.

Run `.venv/bin/python benchmarks/apple/summarize_attention.py` to regenerate
the summary; add `--check-checkpoints` to repeat state comparisons when local
checkpoints exist. Both benchmarks and the summary pass Ruff. Numerical
comparisons above exercise the new benchmark paths; library code is unchanged.
