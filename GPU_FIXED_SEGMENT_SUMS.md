# Fixed-order sums for dynamic GPU grouping

Set `XQL_MLX_FIXED_SEGMENTS=1` to sum float32 groups with dynamic
GPU keys through a fixed tree. The runtime sorts keys stably, gathers
their values, and computes a segmented prefix tree. Only each segment's
endpoint writes its sum. Other lanes write to distinct padding slots,
so the final write has no competing contributions.

The implementation uses standard [MLX](https://github.com/ml-explore/mlx)
array operations and applies to arbitrary dynamic keys and row layouts.
SQL row presence and NULL masks are evaluated through the same grouped
program. Integer counts retain the existing exact bounded implementation.
The optional compact indexed UNION path uses the same reducer for each
branch's duplicate sum. This defines an addition tree for floating
groups, which can differ from the atomic accumulation order.

## Measured behavior

Apple M5 Ultra, MLX 0.32.4, pinned 123.6M Modded-NanoGPT model,
FineWeb tokens, batch 1 and context 1,024, with full SQL validation.
Each training run has twenty updates and averages updates 11–20.
The coordinate-map pass is enabled. The earlier layout and indexed
UNION experiments are disabled.

| Configuration | Warm ms/update |
|---|---:|
| Atomic lookup sums | 592.58 |
| Fixed lookup sums, first run | 596.66 |
| Fixed lookup sums, independent repeat | 602.57 |
| Integrated fixed lookup sums | 604.02 |

All 147 parameter and optimizer arrays matched bit for bit across
the three fixed-order runs. The atomic control and fixed result
exceeded the 3e-5 tolerance on one tied embedding weight, with maximum
absolute difference 4.43e-5. These are distinct floating-point
accumulation policies. The fixed-order policy is opt-in.

The trace selected 75 fixed lookup sums in the training step. One
768-by-1,024 lookup reduction into 50,304 groups took 0.49 ms with
atomic addition and 1.17 ms with the tree in the isolated probe.
Shared metadata and scheduling make this an independent measurement
from the complete training step.

The SQL tests cover repeated and different keys, nonconsecutive
labels, masks, keys that vary per row, all-NULL groups, and empty
filtered results. The implementation also passed isolated integer-valued
sum checks with zero, one, and multiple input positions.
