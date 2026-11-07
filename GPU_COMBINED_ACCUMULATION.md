# Combine symbolic validity and compact embedding accumulation

The symbolic group-validity pass and the existing compact indexed UNION
runtime work well together. The compiler keeps SQL row/NULL validity as
metadata where it can prove a complete domain. The runtime accumulates
lookup-gradient partials in token-sized buffers instead of creating a dense
embedding buffer for each partial. Duplicate reductions complete per branch;
the dense seed, masks and branch order remain in their original positions.
[Symbolic validity](GPU_SYMBOLIC_VALIDITY.md) ·
[Compact indexed accumulation](GPU_INDEXED_ACCUMULATION.md).

## Matched advancing training

Measured November 7, 2026, on Apple M5 Ultra, MLX 0.32.4, FP32 with
`MLX_ENABLE_TF32=0`. The historical 123,568,128-parameter
[Modded-NanoGPT recipe](MODDED_NANOGPT.md) uses real FineWeb tokens, batch 1,
context 1,024, five Muon iterations and the unchanged optimizer schedule.
Every run starts from scratch, advances twenty updates, and averages
updates 11–20. Full SQL validation, coordinate-map reuse and fixed-order
lookup sums remain enabled. The layout experiment is disabled.

| Configuration | Initial warm ms/update | Reverse-order repeat |
|---|---:|---:|
| Symbolic validity, ordinary accumulation | 481.37 | 487.07 |
| Separate compact indexed branches | 429.74 | — |
| Shared compact indexed branches | 417.24 | 413.67 |

The combined shared policy improves warm throughput by 1.15–1.18× against
symbolic validity alone. A later clean production-runtime run measured
416.47 ms/update. All 147 parameter and optimizer arrays match the control
bit for bit in each comparison, including the preceding installed SQL
baseline. Timed updates have zero numerical-array readback bytes and zero
intermediate exports. Checkpoint export occurs after timing.

First-update observations were 7.13/6.94 seconds for ordinary accumulation
and 6.74/6.64 seconds with sharing. Peak process RSS including checkpoint
export was 21.75/21.86 GB versus 21.45/21.45 GB. These serial measurements
do not establish stable compilation or broad memory improvements.

The initial series uses the installed EinFold commit
`3d85cf7527ba3173238079a9fb79fec4862a10a9`. Reverse-order probes use the
feature checkout with the later factorization and value-identity policies
disabled; their numerical checkpoints are also bitwise identical.

## Reproduce

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_SYMBOLIC_MASKS=1 \
  EINFOLD_DENSE_INDEX_MAPS=1 EINFOLD_DENSE_LAYOUT=0 \
  EINFOLD_DENSE_FACTOR_MASKS=0 EINFOLD_DENSE_VALUE_IDENTITIES=0 \
  XQL_MLX_FIXED_SEGMENTS=1 XQL_MLX_INDEXED_UNION=1 \
  XQL_MLX_SHARED_INDEXED_UNION=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/ml/04_modded_nanogpt.py \
  --backend sql_mlx --data-dir ../modded-nanogpt/data/fineweb10B \
  --batch 1 --steps 20 --warmup 10 --validation full --checkpoint \
  --out-dir /tmp/modded-combined
```

Use fresh output directories. Disable both indexed flags for the ordinary
control; retain only `XQL_MLX_INDEXED_UNION=1` for separate compact branches.
This is a flag combination over existing implementations.
[Per-update measurements and checkpoint comparisons](benchmarks/apple/combined-accumulation.json).
