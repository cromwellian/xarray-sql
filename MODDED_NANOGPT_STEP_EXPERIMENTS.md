# Modded-NanoGPT whole-step profiles and compiler experiments

Measured November 6, 2026, on an Apple M5 Ultra using MLX 0.32.4 and
PyTorch 2.15.0. Each batch-1 timing uses twenty updates and averages
updates 11–20. The model has 123,568,128 parameters, context 1,024, and
uses the pinned historical Modded-NanoGPT training recipe with real
FineWeb tokens, Muon, and AdamW. The SQL runs use full row, NULL and
finite-value validation. Timings include forward, backward, optimizer,
and synchronization. Diagnostic graph export and checkpoint I/O are
outside the steady-state measurements. [Recipe and data](MODDED_NANOGPT.md).

## Training times

| Batch 1 policy | Setup | First update | Warm ms/update | Tokens/sec |
|---|---:|---:|---:|---:|
| Generic SQL/MLX FP32 | 1.91 s | 9.78 s | 615.11 | 1,665 |
| Separate compact indexed writes | 2.00 s | 9.50 s | 531.40 | 1,927 |
| Ordered shared indexed write | 1.93 s | 9.40 s | 521.48 | 1,964 |
| Combine token cotangents before duplicate reduction | 1.95 s | 10.20 s | 518.01 | 1,977 |
| Explicit GEMM layouts | 1.94 s | 9.85 s | 614.57 | 1,666 |
| Vocabulary projection BF16 | 2.08 s | 10.67 s | 615.56 | 1,664 |
| PyTorch/MPS FP32 | 1.44 s | 0.30 s | 181.85 | 5,631 |

In a reverse-order repeat, the indexed controls measured **525.54 ms**
with separate writes and **526.92 ms** with sharing. The incremental
warm timing difference is within run-to-run variation. A second layout
and BF16 series measured 606.26 and 605.77 ms, with a subsequent FP32
reference at 607.61 ms. On this workload the ordered shared run reached
2.87× PyTorch's update time, versus 3.38× for generic SQL.

At batch 2, the twelve-update comparison averages updates 7–12. Separate
indexed writes took 667.25 ms and shared writes took 659.68 ms, or
3,105 tokens/sec. PyTorch took 238.06 ms, or 8,603 tokens/sec. This
batch-2 pair covers timing and loss. The state comparisons below use
batch 1. Shared and separate indexed runs agree within 9.54e-7 on
training losses at both batch sizes.

## Shared lookup gradients

The trace proved that 74 partials reference the same ordered token
lookup. One dense branch separates the first partial from the remaining
73. The ordered lowering shares the offset calculation and compact
duplicate-key layout for that run. It applies each branch's masks,
preserves the dense seed and addition sequence, and writes the
accumulated values once. Two flags expose it for comparison.
[Indexed accumulation design](GPU_INDEXED_ACCUMULATION.md).

The token-space experiment applies the masks to token-position values,
adds the 73 contributions, and reduces duplicates once. This changes
the floating-point addition order. It saves 3.47 ms in the single
paired measurement and misses the checkpoint tolerance on one tied
embedding weight. The benchmark driver exposes it as a probe.

| Batch 1 checkpoints after 20 updates | Maximum absolute difference | Elements outside tolerance |
|---|---:|---:|
| Ordered shared vs separate indexed | 2.55e-5 | 0 |
| Ordered shared vs generic SQL | 9.08e-5 | 2 |
| Token sum vs ordered shared | 3.61e-5 | 1 |
| Explicit GEMM layouts vs generic SQL | 9.85e-5 | 1 |
| Vocabulary BF16 vs generic SQL | 3.26e-2 | 73,662,273 |

Each comparison checks all 147 model and optimizer arrays with
absolute and relative tolerances of 3e-5. The isolated failures above
are tied embedding weights. The ordered runtime stays opt-in.

## Full-step graph and allocator profile

A separate diagnostic captures the lazily compiled graph immediately
before update 5 evaluates its loss, validation, weights and optimizer
state. It records ordered contraction axes and dimensions and names
all 149 returned arrays. The liveness calculation uses each value's
last consumer in one valid topological order and keeps returned values
live through its end. Transpose, reshape and broadcast nodes represent logical
layouts. MLX chooses their physical storage and reuses buffers.

| Profile | Logical operations | Logical matmuls | Peak logical live values | MLX peak active GB |
|---|---:|---:|---:|---:|
| Generic SQL | 24,466 | 1,870 | 3,840 | 37.26 |
| Explicit GEMM layouts | 24,466 | 1,870 | 3,840 | 37.26 |
| Separate indexed writes | 26,987 | 1,865 | 4,175 | 15.90 |
| Ordered shared write | 24,624 | 1,865 | 3,749 | 16.28 |

The shared write removes 2,363 logical operations from the indexed
graph, including 145 scatter-sum nodes. The original compact indexed
path accounts for the allocator peak reduction against generic SQL.
The shared path adds a padded destination during its final write.
The profile records active and cached MLX memory separately from
process RSS.

The layout probe selects plain or batched matrix multiplication for
2,472 FP32 contractions. It groups free, batch and contracted axes,
reshapes both operands, and restores SQL output order. MLX's existing
einsum path produces the same operation counts and allocator peak on
this recipe. The experiment stops at operator selection and logical
liveness. Physical buffer scheduling would require allocator or
backend changes.

The generic warm evaluation took about 600 ms with the graph export
outside that interval. The training timer was about 608 ms for the
adjacent uninstrumented update. Cold setup took about 1.9 seconds
before the first update, and accumulated dense lowering accounted for
about 6.16 seconds of the cold run. In a separate Xcode Metal System
Trace, warm GPU work spans lasted 612–618 ms. Each span contained
1,073–1,088 compute interval records, including nested records.
The shader timeline table had no rows. The trace gives encoder
activity, while the DOT report gives graph operations. Per-kernel
durations, physical buffer aliasing, and memory traffic remain
unmeasured.

## Vocabulary projection precision

The BF16 probe requires the recipe's rank-2 vocabulary matrix, rank-3
activation tensor, and exact free and contracted axes. One forward
projection matched. It casts the operands to BF16, then casts the
BF16 result back to FP32 before the loss reductions. The ddx formulas,
other contractions, master weights and optimizer state use FP32.
This test includes conversion costs and approximates the SQL
projection's arithmetic.

On the first update the loss moved from 16.07809 to 16.08018. All 73 raw
ddx gradient arrays passed the elementwise tolerance on that update,
with maximum absolute difference 9.45e-6. Twenty advancing updates
showed much larger differences in model and optimizer state. Whole-step
timings show no clear gain from this precision policy.

## Verification

The local suite with MLX and the pinned upstream PyTorch oracle enabled
passed 876 tests, with 277 optional skips and 102 network-integration
deselections. A focused run with both indexed flags enabled passed 105
tests, including duplicate keys, distinct lookups, NULL masks, attention
and advancing optimizer updates against the pinned oracle.

## Reproduction

Use the pinned EinFold wheel at
`66b8f4584b710b8f38fc75e3af7e0d7e0ada51df` and launch from xarray-sql.
Use fresh output directories for each run.

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_LAYOUT=0 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/apple/profile_modded_step.py \
  --policy reference --profile-dir /tmp/modded-step-profile \
  --backend sql_mlx --data-dir ../modded-nanogpt/data/fineweb10B \
  --batch 1 --steps 6 --warmup 4 --validation full \
  --out-dir /tmp/modded-step-profile-run
```

Omit `--profile-dir` for timing. Use `--steps 20 --warmup 10 --checkpoint`
to reproduce the first timing series and write the final state. Select
`--policy layout` or `--policy bf16` for those probes. Set
`XQL_MLX_INDEXED_UNION=1` for separate indexed writes and add
`XQL_MLX_SHARED_INDEXED_UNION=1` for the ordered shared path.
With both flags, use `--policy token-sum` to test the reordered variant.
Use `--raw-gradients /tmp/modded-first-gradients.npz --steps 1 --warmup 0`
to export the first update's ddx gradients and loss separately.

[Profiler and probes](benchmarks/apple/profile_modded_step.py) ·
[Timing samples, graph counts and numerical checks](benchmarks/apple/modded-step-experiments.json).
