# SQL attention backward lowering and exact group proofs

November 2, 2026. Apple M5 Ultra, 30 CPU/64 GPU cores, 256 GiB,
macOS 27.0.1. Float32, MLX 0.32.4, `MLX_ENABLE_TF32=0`.
This follows the [attention headroom study](GPU_ATTENTION_HEADROOM.md) and
[forward extension](GPU_ATTENTION_FORWARD.md) and
[backward extension](GPU_ATTENTION_BACKWARD.md).

The pipeline remains SQL → [DataFusion](https://datafusion.apache.org/),
the SQL engine → [ddx](https://github.com/xqlsystems/ddx), the SQL
differentiator → [EinFold](https://github.com/xqlsystems/einfold), the compact
tensor optimizer → [MLX](https://github.com/ml-explore/mlx), Apple's GPU
array framework. No model-specific SQL substitution or handwritten Metal
kernel is introduced.

## What is implemented

EinFold retains SUM/MIN/MAX and ordered UNION value/mask/coordinate
provenance. The host can inspect that structure and decline to the original
reduction closure. Two independent group proofs also avoid redundant found
masks for validated integer domains and prove presence from complete,
independent, valid UNION branches. Numerical accumulation, duplicate
contributions and nullable/partial layouts retain their semantics.

The host recognizes eligible no-dropout attention and its **actual
ddx-generated derivative expressions**. A bounded symbolic normalizer checks
both quotient-rule contributions, their coefficient, the key reduction
and the masks. Boolean proofs retain an unknown result for unproved
predicates, and preserve the scope of bound reduction coordinates.
Nonempty domains, compatible axes and masks are required. A query-dependent
cotangent row gate is retained: the proof checks the valid-row derivative
and the zero contribution from absent rows, then masks those cotangents on
the GPU. SQL validity flags are not discarded.

Only after this proof does the host call MLX's existing attention backward
implementation. The public API exposes it through
[`mx.vjp`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vjp.html)
applied to the **single attention primitive**. It does not differentiate
the model, loss, optimizer or SQL program. ddx remains the source of the
derivative contract, including stop-gradient boundaries; unsupported
derivative blocks retain their original expressions.

Sibling backward outputs can be reused under the existing computed-value
LRU byte budget. Eviction clears their owning cache entry; no cross-step
numerical-value cache is added. Array limits also guard the additional
backward outputs. The measured full model selects six primitive backward
calls for Q/K gradients; its value-gradient shape still takes the generic
path. The isolated attention benchmark reuses all Q/K/V outputs. This is
not complete forward/backward scheduling or complete backward fusion.

Zero dropout now removes its mask tables/operators from the normal recipe.
Probability dropout above zero continues on the generic path because the
[MLX attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
has no probability-dropout argument. A logits mask or output dropout would
change the operation and is not substituted.

## Complete training measurements

Every row starts from scratch and completes exactly **100 training updates**
of the **10,745,088-parameter Shakespeare character model**: six layers,
six heads, width 384, vocabulary 65, batch 12/context 256, exact GELU, tied
embeddings, global clipping and AdamW. The original 2,000-iteration
learning-rate schedule is retained. These are training results, not inference
or complete 124M-model measurements.

All variants keep compiled owned GPU state, frozen context, full validation
and discarded inspection outputs. Warm means use updates 2–100. The step
timer includes forward, ddx backward, clipping, optimizer, synchronization,
status and loss reads. Input sampling/binding, evaluation, logging and
checkpoint I/O are excluded. Runs were serial; repeats reverse the order.
These observations are not confidence intervals.

| No-dropout variant | Warm ms/update | First update s | All 100 timed updates s |
|---|---:|---:|---:|
| Reference: group proofs off, generic attention | 48.71 | 0.83 | 5.653 |
| Group proofs only | 38.96 | 0.74 | 4.601 |
| Group proofs + forward-only attention | 39.39 | 0.76 | 4.664 |
| Group proofs + verified backward lowering | 36.83 | 1.01 | 4.659 |
| Selected repeat, run before its reference | 36.58 | 1.03 | 4.649 |
| Reference repeat | 49.88 | 1.01 | 5.945 |
| Final rebuilt pin, bounded sibling cache | 36.51 | 1.07 | 4.681 |

Group proofs alone give **1.250×** warm improvement. Attention backward
adds **1.058×** over that group-only run. Combined improvement is **1.323×**
initially, **1.364×** in the reverse-order repeat; the final-source observation
is **1.366×** against that repeat reference. The corresponding input-plus-step
ratios and all individual samples are retained in the summary/CSV.

Forward-only replacement regressed warm time slightly and is not selected
as the full training optimization. Cold costs matter: adding attention to
group proofs alone made the initial 100 timed updates slightly slower,
4.601→4.659 seconds, despite its warm saving. Compared with the original
reference, the combined total still improved **1.213×**, and **1.279×** in
the repeat. First-update timing varies and is not whole-application startup.

The original **GPU-dropout .2** recipe separately measured:

| Dropout .2 variant | Warm ms/update | First update s | All 100 timed updates s |
|---|---:|---:|---:|
| Reference | 66.75 | 1.26 | 7.866 |
| Group proofs, probability dropout unchanged | 50.97 | 2.41 | 7.461 |

That is **1.309×** warm and **1.054×** including first-step compilation inside
the timer. No built-in attention backward was selected for this recipe.
The difference between dropout-zero and dropout-.2 results is not a
same-recipe speedup claim. No full 2,000-update training run is claimed here.

## Memory, host work and correctness

The initial no-dropout reference's observed MLX peak was 5.28 GB versus
5.91 GB selected; allocator cache ended near 6.46 versus 6.86 GB. Active
memory ended near 135 versus 131 MB. These peaks include evaluation/tracing
and can vary; there is **no overall physical-memory saving claim**.
Computed-cache accounting fell 5.08→3.98 GB in the final source. That is
logical retained-value accounting, including shared references, not measured
physical GPU memory.

The final program records 52 indexed-group domain proofs and 69 complete
UNION presence proofs. Group-related numerical/flag merge dispatches fall
from 146 to 43 for the selected no-dropout run, and direct branch additions
from 1,401 to 455. These counters include preparation/evaluation/tracing;
they are not per-update kernel counts or GPU-time percentages. Numerical
scatter reductions remain 156. The timed loop exports no numerical arrays
and reads back zero numerical-array bytes; one status and one loss scalar
are still read each update. Input binding and Python submission remain.

The rebuilt pinned wheel's Python source matches committed EinFold
`805f50ca6f57b9249815c3d5785e2c78724ef789` byte for byte. The full xarray-sql
suite passes **855 tests**, 277 optional skips and 102 network-integration
deselections, with 83 existing warnings. EinFold Python passes **270 tests**
with two optional skips. Native formatting, Clippy, docs and REUSE pass.
The unchanged Rust `nan_compares_by_its_sign` platform test still fails;
excluding it leaves 72 passing tests and four ignored.

New GPU cases cover full outputs and all Q/K/V gradients, irregular and
rectangular masks, empty/nullable rows, dropout, correlated axes, extreme
constants, changed inputs, shared projection sources, sparse target
gradients, five advancing steps and eviction under a 16 KiB cache. Five
full-model updates with zero dropout and five with GPU dropout .2 both pass
the independent original nanoGPT/PyTorch loss, every raw gradient, weight
and Adam-moment oracle. Oracle exports are outside performance measurements.

All compared 100-update checkpoints pass `atol=rtol=3e-5` across 33 arrays.
Maximum absolute differences are below 2.4e-7 without dropout and 1.4e-7
with dropout. Checkpoints remain local; timing/error text evidence is retained.

## Reproduction and remaining work

Rebuild the pinned dependency with
`.venv/bin/python accelerators/einfold_mlx/build.py --source ../einfold`.
Run the original recipe with selected defaults:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --discard-intermediates --freeze-context \
  --validation full --steps 100 --log-interval 1 --profile-memory \
  --out-dir /tmp/shakespeare-selected
```

Add `--dropout 0` for eligible attention lowering. Use a fresh output
directory for each run. The reference sets `EINFOLD_DENSE_GROUP_PROOFS=0`
and `XQL_MLX_ATTENTION=0`; group-only sets proofs to 1 and attention to 0.
`XQL_MLX_ATTENTION=forward` isolates the forward-only experiment; `all`/`1`
selects proved forward/backward lowering and is the final default. Keep
`MLX_SDPA_VJP_FALLBACK` unset/zero for the built-in backward implementation.
Validation defaults to full; none of these switches weakens it.

[Summary](benchmarks/apple/attention-lowering-summary.json),
[per-update CSV](benchmarks/apple/attention-lowering-steps.csv), and
[summary script](benchmarks/apple/summarize_attention_lowering.py) retain the
ablation results. Regenerate from local run directories with
`--check-checkpoints` when checkpoints are available.
The [paired kernel harness](benchmarks/apple/compare_attention_lowering.py)
compares generic and lowered SQL in alternating order with fixed inputs;
it is distinct from progressing full-model training.

The actual SQL attention replay uses 30 warmups and 16 alternating groups
of ten synchronized calls per variant. Forward + all Q/K/V gradients
improve **1.404→0.692 ms (2.031×)** at 12×6×256×64,
**2.681→1.175 ms (2.281×)** at 12×12×256×64, and
**3.392→1.279 ms (2.653×)** at 1×12×1024×64. Policy validation is omitted
equally in this numerical-kernel comparison; full-model timing retains it.
[Raw replay results](benchmarks/apple/attention-lowering-replay.json) retain
all samples and output errors. They are not full-model speedups.

An additional auxiliary-sharing experiment requested the primitive's saved
forward outputs through a zero-cotangent VJP so the compiler could share
them with backward. Attention-only replay improved 12.7–18.9%, but paired
complete-step replay changed only **35.014→34.728 ms**, **0.8%**, with packed
outputs agreeing within 3.8e-9. Separate 100-update training measured
36.512→36.143 ms; its checkpoint also matches. This small whole-step gain
does not establish a broad benefit or memory improvement, so the runtime
change is not selected. The unapplied
[patch](benchmarks/apple/attention-aux-experiment.patch),
[full-step replay harness](benchmarks/apple/compare_attention_aux.py) and
[replay evidence](benchmarks/apple/attention-aux-step-replay.json) preserve
the experiment; apply it with `git apply --unidiff-zero` before using that
harness. These are
fixed-input replays and a separate trajectory, not the main ablation table.

General consumer-aware matrix packing/layout selection, normalization and
cross-entropy primitive recognition, complete value-gradient reuse, and
joint forward/backward liveness/recomputation scheduling remain. This is a
completed compiler increment, not completion of the entire optimization plan.
