# Normalization and class-loss kernel experiments

These diagnostics test [MLX](https://github.com/ml-explore/mlx) built-in
root mean square (RMS) normalization and cross-entropy operations on the
historical [Modded-NanoGPT recipe](MODDED_NANOGPT.md). SQL and
[ddx](https://github.com/xqlsystems/ddx) still supply the model and derivative
expressions. The prototype proves complete-domain expressions before kernel
selection, retaining the relational fallback for unproved masks or layouts.

The accepted performance improvement is the separate
[numerical-identity pass](GPU_NUMERICAL_VALUE_IDENTITIES.md). The primitive
prototype is retained as an **unapplied patch**, not as a normal runtime
policy: [experimental code and regressions](benchmarks/apple/verified-primitives-experiment.patch).

## Full-model trials

Apple M5 Ultra, MLX 0.32.4, FP32, real FineWeb, 123,568,128 parameters,
batch 1/context 1,024 and full validation. Runs advance twenty updates and
average updates 11–20. These trials use the earlier partial value-identity
prototype, shared compact accumulation and fixed-order lookup sums; they
precede the final indexed-value-sharing gain to 239 ms. Their control is
about 418 ms, not the final installed policy.

| Primitive policy | Warm ms/update | Checkpoint elements outside tolerance |
|---|---:|---:|
| Ordinary operations, partial identities | 417.85 / 418.66 | 0 |
| RMS forward | 406.36 | 12 |
| Cross-entropy forward | 420.73 | 0 |
| Both forwards | 406.87 | 12 |
| Proved RMS adjoints, ordinary forward | 426.13 | 45 |

RMS forward selected 123 normalizations during tracing. The backward proof
uses a bounded rational-polynomial form to verify the quotient-rule terms
before invoking MLX's normalization VJP. It selected 267 adjoints across
the expanded backward regions. These are graph-construction counts, not
distinct executed GPU kernels. Primitive selection changes FP32 evaluation
and addition order. Nine public-query cases cover normalization, class loss
and logsumexp, dense/NULL/filtered rows, nonconsecutive labels and changed
inputs. The focused primitive/model/attention run passed 108 tests; the expanded
nine-case primitive suite passed separately.
The full-size checkpoint results above do not support promoting RMS lowering.

Cross-entropy forward selected once and preserved all 147 state arrays bit
for bit. Its full-model timing does not establish a speedup. A backward
softmax-component prototype declined the actual model's indexed-target mask
proof and selected no components there. That trial measured 419.58 ms and
is not evidence of backward-kernel performance.

## Isolated cross-entropy backward headroom

The [kernel probe](benchmarks/apple/compare_cross_entropy_kernels.py) obtains
real logits from the trained twenty-update checkpoint. It then holds the
`1×1024×50304` FP32 logits and targets fixed and alternates two compiled
loss-and-gradient functions in ABBA order. All logits gradients are
synchronized inside each timer; model forward and optimizer work are excluded.

| Loss and complete logits gradient | Median time |
|---|---:|
| Logsumexp and target lookup | 2.425 ms |
| Built-in fused cross-entropy | 1.162 ms |

The isolated speedup is 2.09×, with identical losses, maximum absolute
gradient difference 1.60e-10 and no elements outside tolerance. Its roughly
1.26 ms saving does not imply a comparable full-model speedup. Proving the
indexed target's mask/cardinality through the relational adjoint remains
necessary before the entire SQL loss backward can select that kernel.
[MLX normalization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rms_norm.html) ·
[Fused cross-entropy implementation](https://github.com/ml-explore/mlx/pull/4520).

The shared model-control helper now exposes logits and accepts SQL's
float64-schema weight exports only when every value round-trips exactly to
float32. Tests check lossless restoration and reject inexact weights; model
gradient oracle tests cover both mathematical and fused attention.

## Reproduce

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python \
  benchmarks/apple/compare_cross_entropy_kernels.py \
  --weights /tmp/trained-model/last.npz \
  --data ../modded-nanogpt/data/fineweb10B/fineweb_train_000001.bin \
  --output /tmp/cross-entropy-kernels.json
```

For the SQL prototype, apply the patch, use fresh output directories and
the ordinary twenty-update training CLI, with
`EINFOLD_DENSE_SHARE_INDEXED=0 EINFOLD_DENSE_SHARE_EXTREMA=0` to retain the
partial-identity control. Select `XQL_MLX_NORMALIZATION=1`,
`XQL_MLX_CROSS_ENTROPY=1`, `XQL_MLX_NORMALIZATION_BACKWARD=1` or
`XQL_MLX_SOFTMAX=1` independently. The patch also includes its public SQL
regressions. Remove the patch after diagnostic runs.
[Raw samples, selection counters and numerical comparisons](benchmarks/apple/primitive-kernel-experiments.json).
