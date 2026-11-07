# What the PyTorch MPS training step dispatches

November 6, 2026. Apple M5 Ultra, macOS 27.0.1, PyTorch 2.15.0
(`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`), and MLX 0.32.4.
Both controls use the pinned historical 123.6M Modded-NanoGPT recipe,
FineWeb tokens, batch 1, context 1,024, and FP32 model and optimizer
state. SQL retains full validation and runs with `MLX_ENABLE_TF32=0`.
[Recipe and controls](MODDED_NANOGPT.md).

## What ran on the GPU

The PyTorch control runs eager autograd. ATen dispatches operators to
MPS, which uses Apple's tuned primitives, MPSGraph, and native Metal
kernels. The operator trace records 73 forward linear calls, 24 forward
batched matrix products, 146 backward matrix products, 48 backward
batched products, and 1,080 Muon matrix products. MPS runtime logs
identify the forward linear path as `mps_linear` and the backward and
Muon products as MPSGraph matrix multiplications. The linear path
describes transposed weight storage to the native MPS primitive using
a descriptor. [PyTorch linear implementation](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/mps/operations/Linear.mm#L34) ·
[PyTorch matrix multiplication implementation](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/LinearAlgebra.mm#L549).

All 12 PyTorch attention forwards selected the math path at this dtype
and shape. The backward trace has one `embedding_dense_backward` call
for the tied embedding lookup. ATen autograd passes the gradient of a
particular forward value to its consumer with that value's tensor
identity. This gives the backend one gather adjoint at that boundary.
The previous SQL profile identified 74 grouped partials derived from
the same token map. [Local PyTorch benchmark driver](benchmarks/ml/_modded_torch.py) ·
[SQL indexed accumulation](GPU_INDEXED_ACCUMULATION.md).

Separate Instruments Metal System Trace runs bracketed the same eight
updates with `mach_absolute_time`. Medians below use updates 5–8.
Compute coverage is the union of the traced Compute intervals. Nested
intervals are counted once for time coverage. A command buffer can
contain multiple kernel launches.

| Warm trace measure | PyTorch MPS | Generic SQL/MLX |
|---|---:|---:|
| Host-observed microstep | 199.4 ms | 619.1 ms |
| Time covered by Compute intervals | 177.2 ms | 598.7 ms |
| Distinct command-buffer IDs in Compute intervals | 6 | 1,018 |

Both shader timeline tables were empty. The traces expose command-buffer
activity, and the PyTorch profiler exposes the selected operations.
Individual Apple MPSGraph shader instructions and per-kernel timings
are unavailable in these captures. The MPS logger repeats command-buffer
durations against several operations, so those durations cannot be summed
into per-operator GPU times.

I tested the command-buffer hypothesis by varying both MLX limits in
separate, uninstrumented 12-update runs. The default and repeated default
averaged 615.5 and 614.1 ms over updates 7–12. Limits of 256 operations
and 400 MB averaged 631.8 ms, and limits of 1,000 operations and
1,000 MB averaged 621.9 ms. The greater observed cost is time inside
the Compute intervals. [MLX command-buffer policy](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp#L573).

## Where the difference appears

The PyTorch phase probe synchronizes after each phase. Medians over
eight warm updates were 23.1 ms for forward, 38.5 ms for backward,
3.0 ms for AdamW, and 138.4 ms for Muon. The barriers alter scheduling,
so the phase values serve as an attribution probe.

An independent replay used the same seeded weights and fixed gradients
for 72 Muon parameters and five iterations. Median warm time was
111.0 ms for a directly expressed MLX tensor graph, 128.8 ms for the
existing SQL Muon stages with full validation, and 138.9 ms for the
same tensor arithmetic on PyTorch MPS. After ten updates, one sampled
matrix of each shape agreed within 2.24e-8 across the three variants.
The replay uses one polynomial evaluation order across backends.
It omits the tied embedding, which is updated by AdamW in the model.

I also ran the compiled SQL loss and ddx gradients with optimizer
stages removed. It kept full gradient validation and held model
weights fixed while advancing FineWeb batches. That graph averaged
473.7 ms over updates 7–12. The ablation changes scheduling and
liveness, so its time should be read independently of the Muon replay.

| Reachable MLX graph slice | Logical nodes | Matmul nodes | Scatter-sum nodes | Gather nodes |
|---|---:|---:|---:|---:|
| Loss alone | 11,242 | 435 | 333 | 503 |
| Loss and 73 raw ddx gradients | 16,777 | 718 | 436 | 790 |
| State outputs beyond those dependencies | 5,748 | 1,152 | 0 | 0 |
| Validation-only dependencies | 1,941 | 0 | 0 | 0 |

The graph was exported before evaluation with named loss, raw gradient,
state and validation roots. Its 24,466 total nodes match the prior
whole-step capture. Counts are MLX primitives. A matmul node can
represent a reduction as well as a large matrix product. ATen operator
counts and MLX graph counts have different granularity.
[Previous full-step capture](MODDED_NANOGPT_STEP_EXPERIMENTS.md).

The existing EinFold coordinate-reshape proof reduced loss-path
gathers from 503 to 266 and scatter sums from 333 to 321. The loss still
reached 435 matmul nodes. The default SQL loss already reached 12 fused
MLX attention nodes, and the loss-plus-gradient graph reached 24. The
earlier shape proof and attention work leave substantial work elsewhere
in the model graph.

## A model-independent compiler path

1. **Make coordinate maps part of the tensor type.** Represent a
   proven SQL coordinate join as an index map with ordered domains,
   multiplicity, row presence, and value validity. A bijective
   affine map can lower to a view or transpose and its adjoint to the
   inverse view. An arbitrary lookup can lower to a gather whose
   adjoint is a scatter-add with duplicate keys. The same proof works
   for a channel reshape, a table permutation, or a learned lookup.
   Retain the relational implementation when the proof fails.
2. **Preserve value identity across prepared SQL and ddx.** Carry
   identifiers for the proven dense values and maps across prepared
   outputs, CTE aliases and derivative boundaries. Lower the dense
   portion to a single typed tensor value graph. Preserve ddx's
   relational transpose rules and attach their adjoints to those
   same values. Keep row and NULL masks and the original FP32
   reduction order unless a separate policy permits reassociation.
   Choose what to save for backward using lifetime and byte cost.
   [ddx relational transpose rules](https://github.com/cromwellian/ddx/blob/953b78ecfc8bce2b1c1a71f0d4c7300444d00cc7/crates/ddx-ad/src/transpose.rs).
3. **Lower a small operator vocabulary and measure the actual result.**
   Use views, gather/scatter, map, reduction, and contraction as
   primitives. Select fusion and layout by dependencies, dimensions,
   masks and memory cost. Compare reachable large contractions,
   gather/scatter work, allocator peak, and synchronized time after
   lowering. Check raw gradients and advancing checkpoints against
   the same tolerance on more than one query family.

A local JAX probe illustrates the intermediate form. For a repeated-key
lookup followed by a contraction and squared loss,
`jax.jit(jax.value_and_grad(...)).lower(...)` emitted one StableHLO
gather, three `dot_general` operations, and one scatter with an addition
combiner and `unique_indices=false`. The MetalJAX plugin executed the
probe on the GPU and its gradients agreed with a NumPy duplicate-key
reference. StableHLO exposes the operation and its indexing semantics
before backend dispatch. This is a small compiler example, with timing
kept separate from the 123.6M comparison.
[JAX lowering API](https://docs.jax.dev/en/latest/jax.stages.html) ·
[MetalJAX backend](https://github.com/eterevsky/metaljax).

## First compiler experiment

The lookup inventory identified cyclic shifts along a split coordinate
axis and maps that were constant along a shared grouping axis. The
EinFold prototype proves both properties from the actual positions.
It replaces a multidimensional lookup with a reshape and a small
one-dimensional lookup per shifted axis. An eligible grouped output
uses the inverse transform. The proof applies independently to any
compatible coordinate map.

With the existing ordered reshape pass as the control, the extension
reduced whole-step scatter-sum nodes from 401 to 225 and logical
operations from 22,801 to 21,696. Warm comparisons moved from
601.2 to 594.0 ms and from 608.3 to 597.8 ms in reverse order.
The final opt-in mode measured 598.8 ms against a 601.5 ms repeat
control. The first update took 8.83 seconds versus 9.71 seconds.
The final checkpoint exceeded tolerance on one tied embedding weight,
with maximum absolute difference 7.82e-5 against the repeated control.
The isolated cyclic-only variant passed the first reference and ran
at 599.1 ms. The extended mode is selected with
`EINFOLD_DENSE_LAYOUT=extended`.

The next compiler boundary is a stable identity for a proven map and
its primal values across the SQL and ddx programs. This experiment
lowers the maps in the current executor. Sharing the value identities
across programs is a separate change.

[EinFold implementation and verification](https://github.com/cromwellian/einfold/blob/7691744e5ecd058575ed877998a75877c8691372/COORDINATE_LAYOUT_OPTIMIZATION.md) ·
[Raw profiling summaries and samples](benchmarks/apple/mps-compiler-evidence.json).
