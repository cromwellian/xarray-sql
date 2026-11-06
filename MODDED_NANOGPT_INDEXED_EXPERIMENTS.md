# Modded-NanoGPT layout, reduction sharing, and indexed accumulation

Measured November 6, 2026, on Apple M5 Ultra with float32 MLX 0.32.4
and PyTorch 2.15.0. All runs use the pinned 123,568,128-parameter
Modded-NanoGPT recipe, genuine FineWeb tokens, context 1,024, one
microbatch per update, and the original Muon and AdamW schedule. SQL
retains full validation. The timing includes sampling, binding, model
forward and backward, optimizer updates, and synchronization. The
serial runs exclude checkpoint I/O from update timers.
[Recipe and timing details](MODDED_NANOGPT.md).

## Individual experiments

Eight advancing updates per variant, with updates 5–8 in the warm mean.

| SQL variant | Warm ms/update | Change vs repeated reference | Final state vs repeated reference |
|---|---:|---:|---|
| Reference, initial | 613.65 | | |
| Reference, repeated | 611.06 | | |
| Proven coordinate reshape | 597.83 | 2.2% less time | Pass, maximum 8.27e-6 |
| Alias sharing for reductions | 629.40 | 3.0% more time | Pass, maximum 2.84e-5 |
| Compact indexed accumulation | 527.42 | 13.7% less time | Pass, maximum 2.40e-5 |

All checkpoint comparisons use the same 147 arrays and absolute and relative
tolerances of 3e-5. Over twenty updates, the layout-only prototype measured
599.21 ms, and the indexed-only prototype measured 527.52 ms. The two
twenty-update references measured 617.85 and 606.77 ms. Each isolated
prototype passed the checkpoint tolerance against the repeated reference.
The combined prototype measured 527.03 ms and failed it on one embedding
weight. Its maximum absolute difference was 5.33e-5 against the first
reference and 4.28e-5 against the repeat.

The coordinate pass proves that an ordered lookup can split an axis by
reshaping and that an independent one-to-one group key can flatten the
result. Its eight-update trace recorded 426 gather and 83 scatter
replacements. The counts describe compilation, and a single compiled
operation can serve many updates. The pass remains opt-in through
`EINFOLD_DENSE_LAYOUT=1`.

The alias experiment keyed a shared numerical result by ordered operands,
their positional axes, masks and reduction extents. It recorded 5,994
sharing hits and reduced contraction builds from 3,797 to 3,077. Warm
training regressed, so this experiment stays outside the runtime.
[Unapplied experiment patch](benchmarks/apple/alias-sum-sharing-experiment.patch).

## Pinned CLI comparison

The final EinFold wheel uses commit
`66b8f4584b710b8f38fc75e3af7e0d7e0ada51df`. The installed Python source
matches the checkout byte for byte. The two runs below passed the explicit
indexed flag values shown in the table. The final default is opt-in.

| Batch 1, twenty updates | Warm ms/update | Warm tokens/sec | First update | All timed updates |
|---|---:|---:|---:|---:|
| Generic SQL, flag 0 | 610.37 | 1,678 | 9.801 s | 21.378 s |
| Indexed SQL, flag 1 | 532.58 | 1,923 | 10.111 s | 20.246 s |

Indexed accumulation reduces warm update time by **12.7%**, a **1.15×**
throughput gain. The cold update was about 0.31 seconds longer in this
pair. Peak process RSS through snapshot export was 27.99 versus 27.52 GB.
Process RSS and the compact buffer size measure different things from
GPU allocator peak. The compiled trace selected 74 indexed partials and
recorded 361 ordinary scatters, down from 436.

The final loss traces agree within 9.54e-7. The full checkpoint comparison
**failed on one tied embedding weight**, with maximum absolute error
4.05e-5 at token 75, channel 493. Its Adam first moments differed by about
2.0e-9. The rebuilt run therefore keeps indexed accumulation opt-in despite
the passing isolated prototype. The combined experiment also remains
opt-in. No tolerance or optimizer epsilon was changed.
[Implementation and proof conditions](GPU_INDEXED_ACCUMULATION.md).

## Batch size

The unmodified SQL and PyTorch controls each completed eight updates at
both batch sizes. Four warm updates per run. RSS below is peak process
resident memory for runs without checkpoint export.

| Backend | Batch | Warm ms/update | Warm tokens/sec | Peak process RSS |
|---|---:|---:|---:|---:|
| Generic SQL/MLX | 1 | 605.96 | 1,690 | 17.50 GB |
| Generic SQL/MLX | 2 | 754.98 | 2,713 | 19.36 GB |
| PyTorch/MPS | 1 | 181.48 | 5,642 | 1.69 GB |
| PyTorch/MPS | 2 | 235.46 | 8,698 | 1.69 GB |

Batch 2 increased throughput **1.61×** on generic SQL and **1.54×** on
PyTorch. At batch 2 the indexed opt-in run measured **678.97 ms/update**
and **3,016 tokens/sec**, about **2.88×** slower than PyTorch at the same
batch. Its eight loss values matched the generic SQL batch-2 run. This
batch comparison covers loss and throughput. The generic batch-2
measurement did not export a checkpoint for a state comparison. Each
batch size changes the effective number of tokens per optimizer update,
so comparisons across batches describe throughput rather than a matched
learning trajectory.

## Verification and reproduction

The final default xarray-sql suite passes 870 tests, with 282 optional
skips and 102 network-integration deselections. The upstream-enabled
opt-in suite passes 19 focused cases. EinFold Python passes 280 tests
with two skips. Rust formatting, Clippy, documentation, and REUSE
pass. The existing Rust NaN-sign test still fails. With that case excluded,
72 Rust tests pass and four are ignored.

The coordinate tests also exposed an existing compact join issue.
For the same two-axis fixture, putting `f.e=m.e` before
`f.pos=m.pos` can duplicate output rows when the lookup depends on
position. The reference reproduced it with the reshape pass disabled.
The shape tests use the shared position equality first and verify the
result against DataFusion. This is a separate join-lowering defect.

From the xarray-sql checkout with the pinned wheel installed, select
indexed accumulation explicitly:

```sh
MLX_ENABLE_TF32=0 XQL_MLX_INDEXED_UNION=1 EINFOLD_DENSE_LAYOUT=0 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/ml/04_modded_nanogpt.py \
  --data-dir ../modded-nanogpt/data/fineweb10B \
  --steps 20 --warmup 10 --validation full --checkpoint --out-dir /tmp/modded-indexed
```

Use a new output directory with `XQL_MLX_INDEXED_UNION=0` for the reference.
For a layout-only comparison set `EINFOLD_DENSE_LAYOUT=1` and leave the indexed
flag at 0. Use `--batch 2 --steps 8 --warmup 4` for the batch comparison.
The archived alias patch applies to EinFold commit `805f50ca6f57b9249815c3d5785e2c78724ef789`
with `git apply --unidiff-zero` and uses `PROBE_ALIAS=1`.

The data file retains all recorded timing/loss samples, trace counters and
the final pinned checkpoint errors. The large snapshots remain in the
local run directories.
[Raw results](benchmarks/apple/modded-indexed-experiments.json).
