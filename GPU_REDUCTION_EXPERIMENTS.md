# Contraction costs, shared values and fixed-order lookup sums

The three experiments use the pinned historical 123.6M Modded-NanoGPT
recipe, real FineWeb tokens, batch 1, context 1,024 and FP32 on Apple
M5 Ultra. SQL validation is full. Coordinate-map reuse is enabled in
both controls and variants. The earlier layout and compact UNION
experiments are disabled. Each timing run has twenty advancing
training updates and averages updates 11–20.
[Recipe and data](MODDED_NANOGPT.md).

## Attribution by shape

The shape probe keeps MLX's simplifier enabled and disables pointwise
fusion in a separate process. MLX 0.32.4 cannot serialize its fused
`Compiled` primitives through the export callback. Both the fused graph
and the companion graph have 1,869 matmul nodes. The probe's tensor
sizes and operation estimates describe the companion graph. It does
not measure physical memory traffic, fused temporary storage, or
individual kernel time. [MLX export callbacks](https://ml-explore.github.io/mlx/build/html/usage/export.html#exporting-with-a-callback) ·
[Native compilation modes](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/compile.h).

The matmul estimate is 3.187 trillion floating-point operations per
step. Of the 1,869 nodes, 321 are scalar or vector contractions with
at least one unit matrix dimension. Their combined estimate is
0.804 billion operations, about 0.025% of the total. The 1,080 products
with Muon's square and wide matrix shapes account for 1.957 trillion
operations, about 61.4%.

| Repeated matmul shape | Count | Estimated operations |
|---|---:|---:|
| 768×768 times 768×3072 | 240 | 870 billion |
| 768×768 times 768×768 | 720 | 652 billion |
| 768×3072 times 3072×768 | 120 | 435 billion |
| 1024×768 times 768×3072 | 46 | 222 billion |
| 1024×768 times 768×768 | 180 | 217 billion |

The unfused logical output sizes sum to 147.6 GB for multiplication,
109.2 GB for casts, and 35.6 GB for grouped scatters. Views and
broadcasts also describe large logical arrays. MLX fuses or aliases
many of these values in ordinary execution. These sizes identify
where to inspect expression expansion, rather than giving a measured
bandwidth total.

The [shape profiler](benchmarks/apple/profile_modded_shapes.py) writes
both primitive metadata and aggregated shape costs. It checks MLX's
version before calling the native compilation-mode API.

## Reduction and contraction value identity

The optional EinFold pass keys a reduction by ordered operand sources,
positional axes, masks, and complete row and grouping domains. It
selects expensive reductions using estimated operations per output
element. The existing byte-bounded computed-value cache controls
retention. [Implementation and proof conditions](https://github.com/cromwellian/einfold/blob/303d65c8bbb16dd4c3994b2bf1b0ab0e29a06232/REDUCTION_VALUE_REUSE.md).

The pass selected 1,081 shared values and reduced contraction builds
from 3,797 to 2,717. The final MLX graph retained 21,873 nodes and
1,869 matmuls. MLX's own optimization already merges those values.

| GPU comparison | Control warm ms | Sharing warm ms | Control first update | Sharing first update |
|---|---:|---:|---:|---:|
| Atomic lookup sums | 592.58 | 590.08 | 7.70 s | 8.40 s |
| Fixed-order lookup sums | 604.02 | 602.31 | 7.97 s | 7.98 s |

The measured differences do not establish a warm GPU speedup.
A separate float64 CPU query computes a 512×512 Gram matrix twice
under different aliases, joins the results, and returns their scalar
sum. Warm medians fell from 2.14 to 1.33 ms in one pair and from 1.99
to 1.42 ms in reverse order, a 1.4–1.6× improvement for that query.
The first pair recorded its timings before a harness teardown error;
the reverse-order pair completed cleanly after that error was fixed.

Select the optional pass with `EINFOLD_DENSE_REDUCTION_VALUES=1`.

## Fixed-order GPU segment reduction

The runtime can use stable key sorting, a segmented prefix tree and
one write per key for float32 groups whose keys are on the GPU.
It uses MLX's standard array operations. Row presence, NULL masks,
and integer counts continue through the SQL program.
[Reducer and numerical policy](GPU_FIXED_SEGMENT_SUMS.md).

| Training run | Warm ms/update |
|---|---:|
| Atomic lookup sums | 592.58 |
| Fixed sums, first process | 596.66 |
| Fixed sums, second process | 602.57 |
| Integrated fixed sums | 604.02 |
| Fixed sums plus reduction sharing | 602.31 |
| Final installed wheel, both options | 600.75 |

All 147 parameter and optimizer arrays matched bit for bit across
independent fixed-sum runs. Sharing on and off also produced identical
arrays with fixed sums enabled. The installed-wheel comparison uses
the same exactness check.

The atomic and fixed-order checkpoints exceeded the existing absolute
and relative tolerances of 3e-5 on one tied embedding weight, with
maximum absolute error 4.43e-5. They use different floating-point
addition orders. The fixed sum costs a small amount of warm time and
provides a reproducible baseline for future compiler comparisons.
Select it with `XQL_MLX_FIXED_SEGMENTS=1`.

## Reproduce

The builder pins EinFold at
`303d65c8bbb16dd4c3994b2bf1b0ab0e29a06232`. Its installed Python
source matches the checkout byte for byte. Run from xarray-sql with
fresh output directories.

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_INDEX_MAPS=1 \
  EINFOLD_DENSE_REDUCTION_VALUES=1 XQL_MLX_FIXED_SEGMENTS=1 \
  EINFOLD_DENSE_LAYOUT=0 XQL_MLX_INDEXED_UNION=0 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/ml/04_modded_nanogpt.py \
  --backend sql_mlx --data-dir ../modded-nanogpt/data/fineweb10B \
  --batch 1 --steps 20 --warmup 10 --validation full --checkpoint \
  --out-dir /tmp/modded-fixed-reductions
```

Toggle the sharing and fixed-sum flags independently for their controls.
For shape attribution, substitute
`benchmarks/apple/profile_modded_shapes.py --shapes-dir /tmp/modded-shapes`
for the benchmark script and use six updates with four warmup updates.
Timings from that diagnostic process include the no-fusion policy.

[Raw timings, shapes and checkpoint comparisons](benchmarks/apple/reduction-experiments.json).

## Verification

The final installed-wheel xarray-sql suite passed 882 tests, with 277
skipped and 102 deselected. EinFold's Python suite passed 286 tests
with two optional skips. Rust formatting, Clippy, documentation and
license checks passed. The full Rust test run reproduced the existing
`nan_compares_by_its_sign` failure; the remaining workspace tests
passed with that test excluded.
