# Independent loss and backward control for the historical 123.6M recipe

The [Modded-NanoGPT recipe](MODDED_NANOGPT.md) now has an independent
[MLX](https://github.com/ml-explore/mlx) autodiff control. It implements the
same tied embeddings, rotary Q/K, FP32 RMS epsilon, causal attention and
squared ReLU. It uses the same 123,568,128 weights, batch 1, context 1,024
and FineWeb tokens as SQL. This is a diagnostic control; SQL continues
to use [ddx](https://github.com/xqlsystems/ddx) for differentiation.

The experiment freezes a checkpoint trained for twenty updates and advances
twenty data batches. It computes loss and every raw parameter gradient,
with finite-value checks. It performs no optimizer update. Inputs are
uploaded before the timer, every returned gradient is synchronized inside
the timer, and checkpoint/gradient exports occur after it. Warm means use
calls 11–20. Measurements use Apple M5 Ultra and MLX 0.32.4, FP32 with
`MLX_ENABLE_TF32=0`.

| Loss and backward computation | Warm ms/call | Peak active MLX GB |
|---|---:|---:|
| Independent MLX, fused attention | 51.95 | 3.57 |
| Independent MLX, mathematical attention | 53.47 | 3.46 |
| SQL with full validation and coordinate-map reuse | 461.43 | 35.04 |

All 73 gradients and the loss agree at 3e-5 absolute and relative
tolerance, with maximum absolute difference 4.77e-7. The 8.88× difference
between SQL and fused MLX describes this fixed-weight backward workload,
not full-model training speed. The similar two native times suggest that
attention-kernel selection alone does not explain the gap.

The tool validates weight shapes, float32 dtype and finiteness, records
weight hashes, and exports SQL gradients in their declared dimension order.
SQL checkpoints exported through a float64 Arrow schema are accepted only
when every weight round-trips exactly to float32; inexact values are rejected.
Its small oracle tests use nonzero output projections and two changed
batches; every gradient is compared against the pinned original PyTorch
model. Fused-attention training requires a supported head width and sequence
length: MLX's short-query vector kernel has no VJP in this release.

## Reproduce

Use a checkpoint from the ordinary twenty-update historical training CLI
with coordinate maps and fixed-order lookup sums enabled, then run:

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_INDEX_MAPS=1 EINFOLD_DENSE_LAYOUT=0 \
  XQL_MLX_INDEXED_UNION=0 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/apple/compare_modded_backward.py \
  --backend native --attention fast --steps 20 --warmup 10 \
  --data ../modded-nanogpt/data/fineweb10B/fineweb_train_000001.bin \
  --weights /tmp/trained-model/last.npz \
  --output /tmp/native-backward.json --gradients /tmp/native-backward.npz
```

Select `--backend sql` and fresh output paths for SQL, or
`--attention math` for the unfused mathematical native control.
The default is the full model; `--preset tiny` is explicitly separate.

[Benchmark tool](benchmarks/apple/compare_modded_backward.py) ·
[Raw timings and numerical comparisons](benchmarks/apple/native-backward.json).
