# Historical speedrun model and optimizer as SQL

[Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) is a
language-model speedrun on the FineWeb web-text dataset. This extension
implements its pinned October 14, 2024 architecture and optimizer equations
as SQL, with a separate adapter for its actual upstream PyTorch classes.

The source is
[`2024-10-14_ModernArch/train_gpt2.py`](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_1_short/2024-10-14_ModernArch/train_gpt2.py),
from revision `4ea6b937337a4889b8cfe3f38a93d120048d8f71`.
The reference adapter verifies its SHA-256 before executing definitions.
The upstream MIT notice is retained beside the benchmark modules.

`_modded_nanogpt.py` generates the model: tied embeddings, root mean square
(RMS) normalization, rotary positions, Q/K normalization, causal attention,
squared ReLU and a separate cross-entropy loss. Defaults are 12 layers,
width 768, six heads and vocabulary 50,304: **123,568,128 parameters**.
The two residual projections start at zero, as in the source. The model
contains no biases, dropout, learned position embeddings or gradient clipping.

It also expresses [Muon](https://kellerjordan.github.io/posts/muon/),
a momentum optimizer that orthogonalizes matrix updates, as ordinary SQL.
The implementation retains Nesterov momentum, transpose of tall matrices,
Frobenius-norm scaling and five Newton–Schulz polynomial iterations. Each
iteration has an explicit device stage so repeated CTE expansion stays
bounded. AdamW updates the tied embedding with the original betas/epsilon;
the learning rate keeps its 5,100-update schedule and 1,450-update cooldown.

`_modded_sql.py` uses the existing owned training session. The loss and
all parameter gradients are [ddx](https://github.com/xqlsystems/ddx) SQL,
lowered by [EinFold](https://github.com/xqlsystems/einfold), the array
optimizer, through [MLX](https://github.com/ml-explore/mlx), Apple's GPU
framework. Parameters and optimizer buffers remain device arrays.
Intermediate optimizer stages remain in the same compiled step; they do
not cross Arrow. Accumulation averages raw gradients across microbatches
and updates weights/moments only at the final microbatch. Full row/NULL/
numerical validation remains the default. CPU execution is float64.

`_modded_torch.py` loads the actual model/Muon definitions, omits the
CUDA/distributed main program and compile decorators, and adapts the
bfloat16 casts to float32. Initial weights and batches are supplied by the
caller, so the control uses the same inputs without copying a second model
implementation. It does not modify the upstream checkout.

Sixteen focused cases cover default parameter counts, the unshortened
schedule, malformed/vocabulary-invalid FineWeb shards, sequential targets,
shard transitions/reset and five advancing updates of two-layer models.
SQL CPU/GPU and one/two-microbatch accumulation compare every loss, raw
gradient, parameter and optimizer moment with upstream PyTorch, then check
evaluation on a new batch. Those tests require an explicitly available
upstream checkout and PyTorch; optional environments skip them.

This is a float32 adaptation for matched local measurements. It changes
precision, including dtype-default RMS epsilon, from the mixed-precision
H100 record. The full-size float32 state comparison is not declared solved:
near-zero embedding gradient differences can be amplified by Adam's 1e-8
epsilon. The accompanying benchmark/evidence extensions expose that limit,
retain a strict checker, and distinguish throughput from time to quality.
No standalone speedup or official speedrun result is claimed by this
recipe extension.
