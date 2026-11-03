# Verified attention derivative-block lowering

[ddx](https://github.com/xqlsystems/ddx), the SQL differentiator, remains
the source of derivatives. [EinFold](https://github.com/xqlsystems/einfold),
the tensor optimizer for SQL, exposes the compact reduction/UNION graph.
This extends the [forward matcher](GPU_ATTENTION_FORWARD.md) to prove
eligible derivative blocks before choosing existing
[MLX](https://github.com/ml-explore/mlx) GPU implementations.

The matcher represents products, quotients and reduced sums symbolically.
It verifies the two softmax quotient-rule contributions, coefficient,
reduction coordinate, mask and operand relationships. Boolean analysis uses
false/true/unknown proof states; unknown is not SQL NULL. Renames respect
local reduction-variable scope. A dynamic cotangent row gate is retained:
the valid-row derivative and zero contribution from absent rows must both
be proved. The backend zeros those cotangents on the GPU.

MLX's public API supplies attention backward through `mx.vjp` applied to
the **single attention primitive**. The SQL model, loss, clipping and
optimizer are not passed to framework autodiff. Unsupported graph shapes,
unproved masks, NULL behavior and dropout retain generic SQL/ddx execution.
SQL row/column flags and full validation remain intact.

Sibling kernel outputs share the existing byte-bounded computed-value LRU.
Eviction releases their cached owner and permits recomputation; there is
no cross-step numerical cache. Additional backward arrays must fit the
existing array bound. Inspection outputs and ordinary Arrow snapshots keep
their contracts. Persistent public state remains owned by the explicit
training session.

`XQL_MLX_ATTENTION=all` or `1` is now the default. `0`/`off` keeps generic
execution and `forward` isolates the earlier forward-only extension.
Float32 and `MLX_ENABLE_TF32=0` remain required. Short sequences or unsupported
primitive shapes fall back. Probability dropout is unchanged: the built-in
attention API has no probability-dropout argument.

On M5 Ultra, the 10,745,088-parameter no-dropout Shakespeare model measured
38.96→36.83 ms warm updates after adding this lowering to group proofs,
1.058×. The initial 100 timed updates, including compilation, changed
4.601→4.659 seconds; this cold cost must be distinguished from warm gains.
Against the preceding generic/group-proof-disabled reference, combined
warm improvement is 1.32–1.37× over independent runs. Full validation stays
enabled; input preparation/evaluation/I/O are outside step timing.

The full model selects six attention backward calls for Q/K gradients;
value gradients still use the generic path there. The isolated attention
case reuses all three outputs. This is not complete backward scheduling.
Measured peak allocation is not reduced reliably, so no overall memory
saving is claimed. Detailed paired measurements are recorded separately.

Validation includes 85 seeded GPU/semantic cases: full outputs and Q/K/V
gradients, changing inputs, nullable/empty rows, dropout, irregular and
rectangular domains, shared projections, sparse labels, extreme constants,
five advancing steps and a 16 KiB cache. The full suite passes 855 tests,
with 277 optional skips and 102 integration deselections. Five full-model
updates in each dropout recipe pass the original
[nanoGPT](https://github.com/karpathy/nanoGPT) PyTorch loss, every raw gradient,
weight and Adam-moment oracle. All compared 100-update checkpoints match
within `atol=rtol=3e-5` across 33 arrays.
