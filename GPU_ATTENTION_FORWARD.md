# Proven dense attention forward lowering

[EinFold](https://github.com/xqlsystems/einfold), the tensor optimizer for
SQL, now exposes compact reduction provenance. The GPU host uses it to
recognize a normalized exponential weighted sum over a Q/K contraction and
select [MLX](https://github.com/ml-explore/mlx), Apple's array framework,
`fast.scaled_dot_product_attention`. Recognition uses expression and axis
relationships, not SQL strings or model/table names.

The matcher verifies the score contraction, shared feature axis, batch/query/
key/value axes, constant scale, and matching maximum/denominator/consumer
domains and masks. Coordinate renames preserve local reduction scope.
Static masks may be causal or irregular, but every query must have an
allowed key. Numerical inputs must be float32. Unsupported fused shapes,
unrepresentable scale constants, value-dependent masks, nullable value
layouts and probability dropout keep the original numerical operations.
SQL row presence and NULL validity remain separate from kernel output.

The implementation accepts at most two batch axes, including the usual
batch/head layout. It converts the actual mask to MLX's causal spelling
only when the complete Boolean matrix agrees; other masks remain explicit.
Thus differing query/key lengths or coordinate labels do not silently change
causal alignment. It does not implement attention-probability dropout.

The ordinary Shakespeare recipe now omits dropout mask tables, RNG sampling
and SQL mask multiplications when `--dropout 0`. Nonzero dropout keeps its
previous behavior. The independent
[nanoGPT](https://github.com/karpathy/nanoGPT) PyTorch oracle also treats
zero dropout as identity. No benchmark-only model patch is needed to run
the eligible SQL recipe.

## Forward-only measurements and limits

Apple M5 Ultra, MLX 0.32.4, float32 with `MLX_ENABLE_TF32=0`. For 100 updates
of the 10,745,088-parameter Shakespeare model, batch 12/context 256, no
dropout and full validation, group-domain proofs alone measured 38.96 ms
warm updates. Adding forward-only attention measured **39.39 ms**, about
1.1% slower. Existing backward expressions still require attention
intermediates, so forward kernel replacement alone is not a training win.
Both final 33-array checkpoints match within `atol=rtol=3e-5`.

This forward extension is useful for eligible queries and as the foundation
for backward lowering. The forward-only review increment is opt-in with
`XQL_MLX_ATTENTION=forward`; it does not enable a training regression by
default. The subsequent backward integration supplies the final default
selection and the complete training report.

Tests compare ordinary SQL/ddx results with the forward specialization for
changing inputs, irregular coordinates/masks, empty/nullable rows,
probability dropout, correlated value/query axes, extreme constants and
rectangular attention. Derivatives continue to be supplied by
[ddx](https://github.com/xqlsystems/ddx), the SQL differentiator; this forward
increment retains their generic execution.
