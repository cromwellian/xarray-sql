# Shared numerical identities across forward and backward execution

The [EinFold](https://github.com/xqlsystems/einfold) tensor compiler now
shares positional numerical values across the complete prepared invocation.
The cache covers pure pointwise expressions, contractions, extrema, proved
indexed sums and ordered UNION values. Consumer coordinates, row/NULL
validity and reduction provenance remain separate. Source values are not
matched by their contents, and identities expire with the invocation.
Numerical storage stays in the byte-bounded LRU and can recompute after
eviction. Operand, reduction and branch orders remain unchanged.

The important new reuse is in indexed reductions: an existing coordinate-map
proof supplies the same retained lookup offsets, while the value key checks
the source, layouts and count bounds. This lets prepared backward regions
reuse the same numerical result while preserving their individual SQL masks.
No model-specific rule, custom GPU kernel or replacement differentiation
engine is added. [ddx](https://github.com/xqlsystems/ddx) still supplies the
SQL derivative program.

The builder pins EinFold at `3108f7a99bec3ac217729da8f517927c36ec4181`.
[Implementation, guards and tests](https://github.com/cromwellian/einfold/blob/3108f7a99bec3ac217729da8f517927c36ec4181/NUMERICAL_VALUE_IDENTITIES.md).

## Full training measurements

November 7, 2026, Apple M5 Ultra, MLX 0.32.4, PyTorch 2.15.0, FP32,
real FineWeb tokens, batch 1/context 1,024, 123,568,128 parameters.
Each independent run starts from scratch and advances twenty updates with
the original optimizer schedule and five Muon iterations. Warm means use
updates 11–20. SQL retains full validation and fixed-order lookup sums.
Checkpoint export occurs after timing. [Recipe](MODDED_NANOGPT.md).

| Policy | Warm ms/update |
|---|---:|
| Symbolic validity alone, current pair | 481.37 / 487.07 |
| Shared compact accumulation | 417.24 / 413.67 |
| Pointwise/contraction identities | 417.85 / 418.66 |
| Complete identities, initial/repeat | 232.93 / 233.98 |
| Complete identities, unchanged host attention lowering | 237.93 |
| Final installed wheel, normal CLI | **238.95** |
| Refreshed PyTorch MPS control | **185.40** |

The installed policy is about **2.0× faster** than symbolic validity alone,
and takes **1.29×** the current PyTorch update time. SQL includes its full
row/NULL/finite checks; PyTorch uses the original eager model and optimizer.
The comparison measures throughput, not validation quality or time to a
speedrun target. Serial measurements are not confidence intervals.

All **147 final parameter and optimizer arrays match bit for bit** against
both the current SQL control and the preceding installed SQL baseline.
This includes repeated runs and the final rebuilt wheel. There are zero
numerical-array readback bytes and zero intermediate exports in warm steps.
First-update time was 5.93 seconds for the installed policy versus
7.13/6.94 seconds in the controls; peak process RSS including checkpoint
export was 20.34 GB versus 21.75/21.86 GB. These are observations from the
recorded runs, not stable startup guarantees.

## Which part matters

| Ablation from complete identities | Warm ms/update |
|---|---:|
| All identity families | 233.98 |
| Indexed identity sharing disabled | 417.95 |
| Extrema identity sharing disabled | 234.95 |

Indexed-reduction sharing accounts for the large gain. The trace reduced
contraction construction from 3,644 to 2,137 and scatter construction from
418 to 197. These are construction counters, not GPU kernel counts.
The large gain also holds with the original host attention matcher.

The independent separable-mask extension selected 123 factorizations.
Initial and reversed pairs measured 417.24→412.58 and 413.67→414.97 ms.
These do not establish an additional warm GPU speedup. The pass remains
available as `EINFOLD_DENSE_FACTOR_MASKS=1`, with unchanged numerical sums.
[Separable-mask proof](https://github.com/cromwellian/einfold/blob/3108f7a99bec3ac217729da8f517927c36ec4181/SEPARABLE_GROUP_VALIDITY.md).

## Backward control and numerical limits

With a fixed twenty-update checkpoint and advancing batches, loss plus all
73 raw gradients now takes **96.04 ms**, versus 340.66 ms before value
sharing. Peak active MLX allocation falls from **23.74 to 5.54 GB**.
Native MLX autodiff took 51.95 ms on the same model and weights. Every raw
gradient and the loss pass 3e-5 absolute/relative tolerance, with maximum
absolute difference 4.77e-7. These are backward computations without an
optimizer, separate from the full-training table.
[Control and timing contract](NATIVE_BACKWARD_CONTROL.md).

The full-size SQL and PyTorch trajectories are not checkpoint-equivalent:
after twenty updates, 17,160 elements exceed the existing 3e-5 tolerance,
with maximum absolute difference 0.00182. This mismatch is already present
in the SQL baseline: the optimized SQL checkpoint is bitwise identical to
it. The throughput comparison does not establish equal convergence.

Built-in normalization/loss forward and backward kernels were also tested.
Their measured benefits and checkpoint failures are documented separately
in [the primitive experiments](GPU_PRIMITIVE_KERNEL_EXPERIMENTS.md). The
rejected code is an unapplied diagnostic patch; it is absent from normal
execution.

## Verification and reproduction

The installed wheel's Python source matches the pinned checkout byte for
byte. The host suite passes 884 tests, with 277 optional skips and 102
integration deselections. EinFold passes 348 Python tests with two skips.
Coverage includes different alias labels and masks, changed inputs,
snapshots, one-byte eviction, empty/nullable groups, duplicate contributions,
attention and advancing optimizer state. Rust formatting, Clippy, Rustdoc
and REUSE pass. The known Rust `nan_compares_by_its_sign` failure remains;
other workspace tests pass when excluded.

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_SYMBOLIC_MASKS=1 \
  EINFOLD_DENSE_FACTOR_MASKS=1 EINFOLD_DENSE_VALUE_IDENTITIES=1 \
  EINFOLD_DENSE_INDEX_MAPS=1 EINFOLD_DENSE_LAYOUT=0 \
  XQL_MLX_INDEXED_UNION=1 XQL_MLX_SHARED_INDEXED_UNION=1 \
  XQL_MLX_FIXED_SEGMENTS=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/ml/04_modded_nanogpt.py \
  --backend sql_mlx --data-dir ../modded-nanogpt/data/fineweb10B \
  --batch 1 --steps 20 --warmup 10 --validation full --checkpoint \
  --out-dir /tmp/modded-value-identities
```

Use a fresh output directory. Set `EINFOLD_DENSE_VALUE_IDENTITIES=0` for
the shared-accumulation control. Independently disable indexed or extrema
identity reuse with `EINFOLD_DENSE_SHARE_INDEXED=0` or
`EINFOLD_DENSE_SHARE_EXTREMA=0`.
[Per-update samples, ablations and numerical comparisons](benchmarks/apple/numerical-value-identities.json).
