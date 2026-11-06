# Compact indexed accumulation for SQL gradients

A tied embedding receives a dense output-head gradient and indexed
contributions from token lookups. The [ddx](https://github.com/xqlsystems/ddx)
SQL differentiator expresses these contributions as a grouped UNION ALL.
[EinFold](https://github.com/xqlsystems/einfold), the array optimizer, retains
the values and key before a grouped scatter. The
[MLX](https://github.com/ml-explore/mlx) GPU runtime uses that metadata in a
compiled training step to avoid materializing a separate dense zero-filled
matrix for every eligible indexed branch.

The runtime proves that the scatter uses an independent lookup key and the
same ordered domain as the UNION output. It groups duplicate keys in a
buffer sized by the number of input positions. It completes each branch's
duplicate sum before adding it to the existing accumulator. The original
branch order, row presence and NULL masks still determine the result.
The lowering applies to float32 values and only when the compact buffer is
smaller. The general UNION and scatter algorithms handle other shapes.

For the 123.6M Modded-NanoGPT recipe at batch 1 and context 1,024, one
eligible partial needs at most a 768 × 1,024 float32 buffer, about 3.15 MB.
The dense branch buffer would have 768 × 50,304 values, about 154.5 MB.
These are buffer sizes before compilation and reuse.

The duplicate-key regression compares compiled gradients with an
independent expected sum, including a masked contribution and values
whose float32 rounding depends on summing duplicates before adding
the dense branch. The upstream-enabled recipe tests also compare raw
gradients and optimizer state through advancing training updates.

Set `XQL_MLX_INDEXED_UNION=1` to opt in. The separate reshape experiment
uses `EINFOLD_DENSE_LAYOUT=1`. The isolated indexed prototype passed the
strict SQL checkpoint tolerance, while the rebuilt implementation missed
it on one embedding weight after twenty updates. The combined layout and
indexed prototype also missed the tolerance on one weight. Both passes
remain opt-in and have independent controls for comparison.

## Sharing the lookup across adjacent branches

Set `XQL_MLX_SHARED_INDEXED_UNION=1` with `XQL_MLX_INDEXED_UNION=1` to
reuse one lookup and one dense write across adjacent compatible branches.
A trace-local cache reuses offsets for the same device key and ordered
coordinate domain. The runtime matches a shared coordinate vector and
offsets after removing axis aliases. A different lookup, layout, or
dense branch ends the run.

The runtime gathers the incoming accumulator at the active keys, sums
each branch's duplicates, and applies that branch's masks. It adds the
masked branch to the compact accumulator in UNION order. The final
write uses distinct destinations for the active keys and padding slots.
This keeps the incoming dense contribution in its original place in
the addition sequence.

On the 123.6M FineWeb workload, one group of 73 indexed partials shared
the write. At batch 1, twenty updates with ten warmup updates measured
531.4 ms per warm update with separate writes and 521.5 ms with sharing.
All 147 checkpoint arrays passed the existing 3e-5 absolute and relative
tolerances against separate indexed writes. Two embedding values exceeded
tolerance against generic SQL, so both controls remain opt-in.
