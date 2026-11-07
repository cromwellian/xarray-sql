# Symbolic row and NULL validity for GPU training

The optional [EinFold](https://github.com/xqlsystems/einfold) tensor compiler
now proves group presence and aggregate validity directly from dimensions
when a mask depends only on group keys. Empty reduced domains give false;
nonempty complete domains preserve that mask. Full-domain counts use
bounded constants. Dynamic reduced-axis masks retain the existing count
calculation. Proofs remain metadata and do not allocate dense proof arrays.
Select this policy with `EINFOLD_DENSE_SYMBOLIC_MASKS=1`.

The builder pins EinFold commit `3d85cf7527ba3173238079a9fb79fec4862a10a9`.
[Proof conditions, tests and implementation](https://github.com/cromwellian/einfold/blob/3d85cf7527ba3173238079a9fb79fec4862a10a9/SYMBOLIC_GROUP_VALIDITY.md).
SQL, [ddx](https://github.com/xqlsystems/ddx) derivative formulas, numerical
contractions and full validation remain in the training program.

## Advancing training

Apple M5 Ultra, [MLX](https://github.com/ml-explore/mlx) 0.32.4, FP32,
123,568,128 parameters, real FineWeb tokens, batch 1 and context 1,024.
Both arms enable coordinate-map reuse and fixed-order lookup sums. Layout
and compact indexed UNION experiments are disabled. Each run starts from
scratch, advances twenty updates, and averages updates 11–20.
[Historical recipe](MODDED_NANOGPT.md).

| Full validation | Control | Symbolic validity |
|---|---:|---:|
| Warm ms/update | 602.30 | 483.11 |
| First update | 7.97 s | 7.13 s |
| Peak process RSS including checkpoint export | 23.00 GB | 21.95 GB |

This paired measurement improves throughput by **1.247×**. All 147 final
parameter and optimizer arrays match bit for bit. The earlier installed
fixed-order baseline, at 600.75 ms/update, also matches exactly. These are
serial runs, not confidence intervals; first-update observations include
compilation and should not be treated as stable startup guarantees.

The rebuilt pinned wheel matches the EinFold Python source byte for byte.
An independent twenty-update run through the normal benchmark CLI measured
**481.23 ms/update**, again with all 147 final state arrays bitwise equal
to the paired control. No source overlay or generated-kernel probe was used.

The separate [native backward control](NATIVE_BACKWARD_CONTROL.md) holds
trained weights fixed while batches advance. Symbolic validity lowered
its SQL time from 461.43 to 340.66 ms/call and peak active MLX allocation
from 35.04 to 23.74 GB. All 73 gradients remain within 3e-5 absolute and
relative tolerance, with maximum absolute difference 1.12e-8 against SQL.
Native MLX autodiff took 51.95 ms. These are loss/backward computations,
without optimizer updates, and are separate from full training timings.

## Other investigated compiler changes

A broader Boolean ANY rewrite for the remaining count-positive predicates
took about 791 ms for backward and was removed. It changes graph boundaries
and does not offer a measured benefit on this workload.

The [generic reduction probe](benchmarks/apple/probe_generated_reductions.py)
generates a scalar expression from ordinary arithmetic tensor nodes and
combines it with a parallel row-reduction template through MLX's Metal
kernel compiler. Eligibility depends on dimensions, unmasked FP32 inputs
and supported expressions; there are no NanoGPT operator names or shapes
in its rules. The prototype selected 46 reductions during tracing. Full
training measured 485.24 ms/update against 483.11 ms for ordinary MLX,
and seven final checkpoint elements exceeded the existing tolerance;
maximum absolute difference was 1.77e-4. It changes the FP32 addition tree.
It remains an explicitly invoked diagnostic and is not installed into the
normal runtime. These results do not support promoting it.

An audit of ddx's relational transpose found that it already accumulates
pending column cotangents and shares materialized regions with more than
two input slots. See [transpose implementation](../ddx/crates/ddx-ad/src/transpose.rs).
No new differentiation pass was added. Carrying one numerical identity
through every relational region is a broader compiler change that remains
open; the present increment addresses redundant semantic bookkeeping.

## Verification and reproduction

With symbolic masks enabled, the local xarray-sql suite passed 884 tests,
with 277 optional skips and 102 integration deselections. EinFold's Python
suite passed 334 tests with two skips. Coverage includes empty domains,
NULL masks depending on group or reduced axes, duplicates, COUNT/SUM/AVG,
changed values, signed zeros, independent model gradients and optimizer
updates. Rust formatting, Clippy, Rustdoc and REUSE pass. The existing Rust
`nan_compares_by_its_sign` test fails; the remaining workspace tests pass
with it excluded.

```sh
MLX_ENABLE_TF32=0 EINFOLD_DENSE_INDEX_MAPS=1 \
  EINFOLD_DENSE_SYMBOLIC_MASKS=1 XQL_MLX_FIXED_SEGMENTS=1 \
  EINFOLD_DENSE_LAYOUT=0 XQL_MLX_INDEXED_UNION=0 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/ml/04_modded_nanogpt.py \
  --backend sql_mlx --data-dir ../modded-nanogpt/data/fineweb10B \
  --batch 1 --steps 20 --warmup 10 --validation full --checkpoint \
  --out-dir /tmp/modded-symbolic
```

Use a fresh directory and `EINFOLD_DENSE_SYMBOLIC_MASKS=0` for the control.
The generated-kernel diagnostic substitutes
`benchmarks/apple/probe_generated_reductions.py` for the training script.
[Per-update measurements and numerical checks](benchmarks/apple/symbolic-validity.json).
