# Reusing coordinate maps across SQL and ddx stages

This experiment carries one ordered lookup and its gathered values across
prepared SQL statements in the same training invocation. The existing
[ddx](https://github.com/xqlsystems/ddx) differentiator supplies the primal
and adjoint plans. [EinFold](https://github.com/xqlsystems/einfold), the
SQL tensor optimizer, shares the numeric values underneath axis aliases
and proves when a gather followed by a lookup in the same domain can
reuse the original offsets. Each statement retains its row and NULL
masks and original group contributions.
[EinFold implementation and proof](https://github.com/cromwellian/einfold/blob/bdde066d30a79f1f8db80633d7b723dedcb7086e/COORDINATE_MAP_REUSE.md).

## Matched FineWeb runs

Apple M5 Ultra, MLX 0.32.4, FP32, batch 1, context 1,024, and
123,568,128 parameters. The runs use the pinned Modded-NanoGPT model,
original optimizer schedule, and real FineWeb tokens. Full SQL
validation runs for every update. The prior layout and indexed UNION
experiments were disabled. Both arms use the same native EinFold
library, with the Python source loaded from the respective checkouts
for the initial comparisons. [Recipe and inputs](MODDED_NANOGPT.md).

| Run order | Control warm ms | Reuse warm ms | Control first update | Reuse first update | Control peak process RSS | Reuse peak process RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control first | 607.94 | 589.41 | 9.86 s | 7.76 s | 27.68 GB | 22.98 GB |
| Reuse first | 605.88 | 591.42 | 9.96 s | 7.88 s | 27.86 GB | 22.71 GB |

The warm average uses updates 11–20. Warm time fell by 2.4–3.0%, and the
first update fell by about 2.1 seconds. Peak process RSS includes
checkpoint export. Checkpoint export occurs after all timed updates.

The built wheel's Python source matches the pinned commit byte for byte.
An eight-update run through the normal benchmark CLI averaged 595.72 ms
with reuse and 608.54 ms with it disabled over updates 5–8. The two
runs reported identical training loss values at all eight updates.

A separate graph capture recorded 24,466 → 21,873 logical operations
and 790 → 201 gather nodes. The trace reported 547 exact map cache
hits, 75 elided round trips and 956 selections of shared gather values.
A stage-level probe observed 622 uses of a map after its first
prepared statement. This included 79 uses of a saved forward map in
the tied embedding adjoint. The counts describe the compiled graph
and its construction.

The offsets and round-trip probe with gather-value sharing disabled
took 615.78 ms per warm update. Keeping map and gather-value sharing
while retaining round trips took 592.87 ms. The larger warm gain comes
from sharing the gathered tensors.

## Numerical checks

Each checkpoint comparison covers 147 parameter and optimizer arrays
with absolute and relative tolerances of 3e-5.

| Comparison | Maximum absolute error | Elements outside tolerance |
|---|---:|---:|
| Reuse versus matched control, first pair | 1.81e-5 | 0 |
| Reuse versus matched control, reverse-order pair | 4.79e-5 | 1 |
| The two controls | 2.40e-5 | 0 |
| Offsets and round trips alone versus first control | 5.35e-5 | 1 |
| Map and gather values with round trips versus first control | 5.88e-5 | 1 |

The misses occur in the tied embedding weights. The enabled experiment
passes the smaller upstream-oracle tests, including the duplicate-key
test with nonconsecutive labels, masked contributions, and changed
token inputs through the same compiled step. The two large comparisons
support keeping the pass opt-in.

The installed-wheel suite with map reuse, MLX, and the pinned upstream
PyTorch oracle enabled passed 877 tests. There were 277 optional skips
and 102 network-integration deselections.

## Reproduce

Build the pinned wheel from a clean local EinFold checkout at
`bdde066d30a79f1f8db80633d7b723dedcb7086e` using the
[wheel builder](accelerators/einfold_mlx/README.md). Run the normal
benchmark entry point from xarray-sql with fresh output directories.

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_LAYOUT=0 XQL_MLX_INDEXED_UNION=0 \
  EINFOLD_DENSE_INDEX_MAPS=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/ml/04_modded_nanogpt.py \
  --backend sql_mlx --data-dir ../modded-nanogpt/data/fineweb10B \
  --batch 1 --steps 20 --warmup 10 --validation full --checkpoint \
  --out-dir /tmp/modded-coordinate-reuse
```

Repeat with `EINFOLD_DENSE_INDEX_MAPS=0` for the control. Keep the
reproduction runs serial for timing. [Raw samples and checks](benchmarks/apple/coordinate-map-reuse.json).
