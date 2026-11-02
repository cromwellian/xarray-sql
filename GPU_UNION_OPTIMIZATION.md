# Identity-layout GPU accumulation

The compact SQL backend uses [EinFold](https://github.com/xqlsystems/einfold)
to retain array axes between queries and [MLX](https://github.com/ml-explore/mlx)
to execute GPU arithmetic. [ddx](https://github.com/xqlsystems/ddx), the SQL
automatic differentiator, emits `UNION ALL` branches to accumulate gradients.
Each grouped branch has one value per coordinate; duplicate branches still
contribute separately.

Previously every GPU branch used scatter-add, an indexed accumulation,
even when its ordered coordinate domains exactly matched the output.
The CPU path already recognized that identity mapping. The GPU path now
uses ordinary elementwise addition in the same case. Different ordering,
partial overlap and different domains retain scatter-add.

The branch order, initial zero, row-presence and NULL masks are retained.
There is no new kernel, dtype change, host numerical readback or rewritten
gradient formula. Coordinate comparisons use existing host metadata during
compilation. `XQL_MLX_DIRECT_UNION=0` selects the original scatter path;
the identity path is enabled by default.

## Training measurement

Apple M5 Ultra, MLX 0.32.4, float32, TF32 disabled, November 2, 2026.
The model is the **10,745,088-parameter Shakespeare character model**:
six layers/heads, width 384, batch 12, context 256, dropout .2,
exact GELU and clipped AdamW. These are complete training updates, not
inference. Each run starts from scratch and executes exactly 100 updates
without shortening the 2,000-iteration learning-rate schedule.

Boolean/presence proof extensions were disabled to isolate layout dispatch.
GPU dropout and full validation were enabled. Warm means use updates 2–100,
including forward, backward, optimizer and device synchronization; input
preparation, initial evaluation, logging and checkpoint I/O are excluded.

| Metric | Scatter reference | Identity dispatch |
|---|---:|---:|
| Warm update | 87.56 ms | 67.52 ms |
| First update, including compilation | 1.33 s | 3.45 s |
| All 100 timed updates | 10.00 s | 10.14 s |
| Observed MLX peak allocation | 4.36 GB | 4.36 GB |

The observed warm speedup is **1.297×**. The first-step cost makes the
100-update total slightly slower in this pair; it amortizes after roughly
108 total updates. Cold timings varied in subsequent compiler ablations,
so this is an observation rather than a guaranteed startup penalty. No
physical-memory reduction is claimed.

The optimized full-run counters recorded 1,401 direct additions and 180
remaining union scatter additions. A separately instrumented reference
recorded 1,581 union scatter additions. The 156 dependent-group scatter
reductions are unchanged. These counters include initial evaluation and
tracing; they are not kernel counts per warm update. Every timed update
retained zero numerical-array readback bytes and zero intermediate exports.

## Correctness and reproduction

The original pinned EinFold dependency passes 53 focused SQL/layout tests.
Eight new cases compare both dispatch modes with DataFusion SQL across
identical, permuted and partial domains, NULL groups, duplicated branches,
changed inputs and retained Arrow snapshots. All 33 weight/moment arrays
after 100 updates agree within 3e-5; maximum absolute difference is 2.39e-7.

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_PREDICATES=off XQL_MLX_DIRECT_UNION=0 \
  PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --discard-intermediates \
  --validation full --freeze-context --steps 100 --log-interval 1 \
  --profile-memory --out-dir /tmp/union-reference
```

Repeat with `XQL_MLX_DIRECT_UNION=1` and a fresh output directory.
[Summary](benchmarks/apple/union-layout-summary.json),
[per-update measurements](benchmarks/apple/union-layout-steps.csv),
[pinned-dependency test log](benchmarks/apple/union-pinned-base-tests.log).
Broader predicate/compiler experiments are reported separately.
