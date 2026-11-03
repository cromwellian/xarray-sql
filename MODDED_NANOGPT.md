# Historical Modded-NanoGPT in SQL

[Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) is a
language-model training speedrun: reach cross-entropy validation loss 3.28
on a fixed FineWeb validation prefix using eight NVIDIA H100 GPUs.
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) is a web-text
dataset; this benchmark consumes the upstream GPT-2-tokenized binary shards.
It does not compare Shakespeare training with that task.

`benchmarks/ml/04_modded_nanogpt.py` implements the **October 14, 2024
ModernArch recipe**, with **123,568,128 trainable parameters**, as SQL.
The source is pinned to repository revision
`4ea6b937337a4889b8cfe3f38a93d120048d8f71`, path
[`records/track_1_short/2024-10-14_ModernArch/train_gpt2.py`](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_1_short/2024-10-14_ModernArch/train_gpt2.py).
The PyTorch control verifies that file's SHA-256 before loading its definitions.
The upstream MIT notice is retained in
[`MODDED_NANOGPT_LICENSE.txt`](benchmarks/ml/MODDED_NANOGPT_LICENSE.txt).

This is a **single-device float32 port**, not the current record recipe or
an official speedrun entry. Short throughput runs and validation-to-target
runs are separate modes. A short run never claims to reach the target.
The numerical comparison boundaries are described below.

## Model and training contract

| Component | Historical recipe and SQL expression |
|---|---|
| Transformer | 12 layers, width 768, six heads of width 128, no biases/dropout |
| Embeddings | 50,304 padded GPT-2 tokens, tied input/output weights; no learned positions |
| Normalization | Root mean square (RMS) normalization, per-position averages of squared values; Q/K normalized after rotation |
| Positions | Rotary embeddings: paired feature lookups and sine/cosine products; causal attention |
| Feed-forward | Width 3,072, squared ReLU (`CASE WHEN x>0 THEN x*x ELSE 0 END`) |
| Initialization | Linear weights uniform within ±1/√fan-in; attention and MLP residual projections zero |
| Loss | Mean next-token cross entropy, including padded vocabulary logits |
| Hidden matrices | [Muon](https://kellerjordan.github.io/posts/muon/), a momentum optimizer with matrix orthogonalization; momentum/Nesterov 0.95, five Newton–Schulz polynomial iterations, coefficients 3.4445, −4.775, 2.0315; matrix norm epsilon 1e-7; transpose tall matrices, then scale by √max(rows, columns) |
| Tied embedding | AdamW, learning rate 0.0036, betas 0.9/0.95, epsilon 1e-8, weight decay zero |
| Muon learning rate | 0.1 times the embedding learning rate |
| Schedule | 5,100 updates; constant learning rate, then linear cooldown over the last 1,450; no gradient clipping |
| Historical batch | 512 sequences × context 1,024 = 524,288 tokens per optimizer update |
| Local defaults | Same model/context, batch **1**, accumulation **1**, 20 updates; throughput uses the last ten updates |

[DataFusion](https://datafusion.apache.org/), the SQL engine, plans the model.
[ddx](https://github.com/xqlsystems/ddx), the SQL differentiator, computes
all parameter gradients from that loss. [EinFold](https://github.com/xqlsystems/einfold),
the array optimizer, executes its numerical plans through
[MLX](https://github.com/ml-explore/mlx), Apple's GPU array framework.
All Muon/AdamW arithmetic is ordinary SQL after the shared ddx backward.
No Python model or whole-model MLX autodiff substitutes for the SQL model.

Muon iterations use explicit intermediate SQL stages in the same compiled
GPU step. This bounds repeated CTE expansion; it does not export the
intermediate matrices to Arrow. Owned weights, momentum and Adam moments
remain on the device. Accumulation adds gradients on the device and runs
the optimizer only on the last microbatch. Validation defaults to `full`.
`--validation state` checks loss and published state; `none` trusts the
numerical/row/NULL contracts and omits runtime checks. Input binding still
validates shapes, domains and values in either mode.
Accumulation uses two prepared graphs, with catalog-generation checks;
single-microbatch compiled training freezes the context.

## Setup and matched throughput

Use the existing checkout environment with the
[local EinFold build](accelerators/einfold_mlx/README.md). Avoid environment
synchronization that replaces those local wheels. The PyTorch control
additionally requires PyTorch in its interpreter. The SQL-only path does
not require it. Keep the upstream checkout beside xarray-sql, or provide
`--upstream /path/to/modded-nanogpt` for controls/oracles.

```sh
git clone https://github.com/KellerJordan/modded-nanogpt.git ../modded-nanogpt
git -C ../modded-nanogpt checkout --detach 4ea6b937337a4889b8cfe3f38a93d120048d8f71

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/04_modded_nanogpt.py \
  --data-dir ../modded-nanogpt/data/fineweb10B --download-data \
  --steps 20 --warmup 10 --checkpoint --out-dir /tmp/modded-sql
```

The download is explicit: one training shard and one validation shard,
about 200 MB each. Existing files are reused. Every shard's magic/version,
declared length and file size are checked; sampled token IDs must be in
the GPT-2/model vocabulary. Input tokens are contiguous and targets are
shifted by one. Training and validation have independent streams.

Run the same arguments with a PyTorch-capable interpreter:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" python benchmarks/ml/04_modded_nanogpt.py \
  --backend torch_mps --data-dir ../modded-nanogpt/data/fineweb10B \
  --steps 20 --warmup 10 --checkpoint --out-dir /tmp/modded-torch

.venv/bin/python benchmarks/apple/compare_modded_nanogpt.py \
  /tmp/modded-sql /tmp/modded-torch --checkpoints --output /tmp/modded-comparison.json
```

Use a fresh output directory for every run. **`--steps 5 --warmup 1` performs
exactly five optimizer updates**, with the same 5,100-update schedule.
It does not initialize the model again between updates. `--warmup` only
selects which advancing updates are excluded from the warm mean.

The comparison script rejects different model/configuration, initial-state
hashes, ordered shard hashes, source files, precision, update counts or
hardware identifiers. It retains every timing sample and per-array
checkpoint error. `--require-state-match` makes the strict checkpoint
comparison fail the command, after writing the report. A reported timing
ratio by itself does not imply checkpoint parity.

`sql_cpu` is a float64 diagnostic path; `torch_cpu` and `torch_cuda` are
additional eager float32 controls. `--synthetic` explicitly selects random
throughput data and cannot be combined with FineWeb quality evaluation.
`--preset tiny --synthetic --steps 5 --warmup 1 --verify cpu` checks every
loss, raw gradient, weight and moment against upstream PyTorch, outside
timers. This requires a PyTorch-capable interpreter. Full-model strict
state checking currently exposes the numerical issue described below.

## Validation-to-target runs

`--mode quality` defaults to the historical validation budget of
**10 × 2²⁰ tokens**, always from the same validation prefix. It evaluates
before training, every 125 updates and at the final requested update.
`--target-loss 3.28` records the first successful evaluation's accumulated
training time. The report also retains setup, cold update, evaluation and
elapsed times since `main()`, excluding interpreter/module startup.
Compilation is included in the cold update;
validation and checkpoint I/O are excluded from accumulated training time.
The old upstream log discarded its first ten advancing updates from its
training timer; this trainer retains them in its total and excludes them
only from the warm mean.

For a bounded validation smoke run, append
`--mode quality --val-tokens 1024 --eval-every 5 --steps 5 --warmup 1`.
That checks evaluation plumbing, not the official validation budget.

To preserve the historical effective batch on one device, use
`--batch 1 --accumulation 512`, or another pair whose product is 512.
The full recipe uses `--steps 5100 --warmup 10 --mode quality` with the
default validation budget. At that batch, at least 27 training shards are
needed; `--download-data --download-train-shards 27` fetches those explicitly.
Quality mode rejects an insufficient training stream instead of silently
wrapping it. This complete run has **not** been performed or validated.
`--checkpoint` exports final state for inspection/comparison; resume is
not implemented in this new benchmark.

## Precision and comparison boundaries

The control loads the actual historical model and Muon classes. An AST
adapter omits their CUDA/distributed main program and compile decorators,
changes rotary/orthogonalization bfloat16 casts to float32, loads identical
NumPy-initialized weights, and uses eager PyTorch on the selected device.
It runs the original model/autograd and optimizer equations. Source files
in the upstream checkout are not edited.

Both local GPU paths use float32. The record used CUDA bfloat16 activations,
float32 parameters/Adam and hardware-specific matmuls. In particular,
PyTorch's dtype-default RMS epsilon changes with precision: the float32
port uses 2⁻²³ throughout, while historical bfloat16 Q/K normalization used
2⁻⁷. Rotary constants and Muon arithmetic are also no longer quantized to
bfloat16. Distributed reduction order, kernels and graph capture differ.
The local stream advances continuously rather than replaying the original
script's prefetched first batch after its loader reset. These differences
are explicit; no bitwise reproduction or historical time-to-quality result
is claimed.

The strict full-model float32 state oracle currently fails: tiny gradient
differences near zero can become materially larger Adam weight differences
because epsilon is 1e-8. The benchmark retains that epsilon and the strict
3e-5 checker. Tests cover small models/accumulation, and the performance
report records the full-size mismatch rather than declaring it correct.
Resolving this is required before claiming strict full-model training parity.
