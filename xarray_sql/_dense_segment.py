"""Fixed-order GPU sums for dynamic SQL grouping keys."""

from __future__ import annotations

import mlx.core as mx


def _sum(data, indexes, size):
    rows, length = data.shape
    result = mx.zeros((rows, size + length), dtype=data.dtype)
    if not length:
        return result
    keys = mx.broadcast_to(indexes.astype(mx.int64), data.shape)
    # Stable sorting preserves the input occurrence order inside each key.
    order = mx.argsort(keys, axis=-1)
    keys = mx.take_along_axis(keys, order, axis=-1)
    # SQL accumulation starts at positive zero, including singleton groups.
    values = mx.take_along_axis(data, order, axis=-1) + 0
    distance = 1
    while distance < length:
        same = keys[:, distance:] == keys[:, :-distance]
        values = mx.concatenate(
            [
                values[:, :distance],
                values[:, distance:] + mx.where(same, values[:, :-distance], 0),
            ],
            axis=1,
        )
        distance *= 2
    ends = mx.concatenate(
        [
            keys[:, :-1] != keys[:, 1:],
            mx.ones((rows, 1), dtype=mx.bool_),
        ],
        axis=1,
    )
    # One endpoint writes each group's sum. Other lanes get distinct padding
    # destinations, so the write itself has no competing contributions.
    destinations = mx.where(
        ends, keys, size + mx.arange(length, dtype=mx.int64)
    )
    return mx.put_along_axis(result, destinations, values, axis=1)[:, :size]
