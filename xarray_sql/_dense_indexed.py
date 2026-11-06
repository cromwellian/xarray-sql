"""Accumulate compact indexed partials into an ordered GPU UNION sum."""

from __future__ import annotations

import math
from itertools import groupby

import mlx.core as mx
import numpy as np
from einfold.dense_program import _align


def _partial(part):
    reduction = part.value.reduction
    if reduction is None or reduction.kind != "sum" or reduction.repeated != 1:
        return None
    indexed = getattr(reduction.value, "indexed", None)
    if indexed is None:
        return None
    key = indexed.key
    output = reduction.value.axes
    size = math.prod(len(a) for a in key.axes)
    # Keep the grouping independent, and require the same ordered domain
    # used by the original device scatter. A compact buffer must be smaller.
    if (
        len(output) != 2
        or not key.axes
        or indexed.local != (output[0], *key.axes)
        or output[0] in key.axes
        or not 0 < size < len(output[1]) <= np.iinfo(np.int32).max
        or key.domain is None
        or not np.array_equal(key.domain.values, output[1].values)
        or set(output) != set(part.axes)
        or len(reduction.row_axes) != 2
        or set(reduction.row_axes) != set(output)
        or set(reduction.kept) != set(output)
        or reduction.factors
        not in {(reduction.value,), (reduction.value, reduction.mask)}
        or not set(part.mask.axes).issubset(output)
        or not set(reduction.mask.axes).issubset(output)
    ):
        return None
    return indexed, output, reduction.mask


def _coalesce(indexes):
    keys = indexes.reshape(-1).astype(mx.int32)
    order = mx.argsort(keys)
    sorted_keys = keys[order]
    starts = mx.concatenate(
        [mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]]
    )
    group = mx.cumsum(starts.astype(mx.int32)) - 1
    compact_keys = mx.zeros_like(keys).at[group].maximum(sorted_keys)
    # Sort the key metadata only. Accumulate values in their original layout.
    slots = mx.zeros_like(keys).at[order].add(group)
    active = mx.arange(keys.size) <= group[-1]
    return compact_keys, slots, active


def _mask_at(runtime, tensor, axes, keys):
    mask = runtime._mx(_align(tensor.get(), tensor.axes, axes))
    return mx.take(mask, keys, axis=1) if axes[1] in tensor.axes else mask


def _unaliased(node):
    while node.kind == "rename":
        node = node.args[0]
    return node


def _shared_key(item):
    part, candidate = item
    if candidate is None:
        return None
    indexed, output, _ = candidate
    key = _unaliased(indexed.key)
    # A lookup of a single coordinate vector depends only on that vector
    # and its ordered offsets. Equal output sizes alone prove nothing.
    if key.kind != "take" or len(key.args[0].axes) != 1:
        return None
    _, offsets, at, _ = key.value
    if at != 0:
        return None
    return (
        _unaliased(key.args[0]),
        id(offsets),
        tuple(len(a) for a in indexed.key.axes),
        tuple(part.axes.index(a) for a in output),
    )


def _compact_part(runtime, part, candidate, keys, slots, active, rows):
    indexed, output, mask = candidate
    data = indexed.value.get()
    if not runtime.owns(data) or data.dtype != mx.float32:
        return None
    data = runtime.broadcast(
        data,
        indexed.value.axes,
        indexed.local,
        tuple(len(a) for a in indexed.local),
    ).reshape(len(output[0]), -1)
    # Reduce duplicates and apply SQL masks separately for each branch.
    partial = mx.zeros((len(output[0]), keys.size), dtype=mx.float32)
    partial = partial.at[rows, slots[None, :]].add(data)
    partial = partial * _mask_at(runtime, mask, output, keys)
    valid = active & _mask_at(runtime, part.mask, output, keys)
    return mx.where(valid, partial, 0)


def _add_indexed(runtime, result, items):
    part, candidate = items[0]
    indexed, output, _ = candidate
    key = indexed.key.get()
    if not runtime.owns(key):
        return None
    indexes, _ = runtime.indexes(output[1].values, key)
    keys, slots, active = _coalesce(indexes)
    rows = mx.arange(len(output[0]))[:, None]
    permutation = tuple(part.axes.index(a) for a in output)
    accumulated = result.transpose(permutation)
    if len(items) == 1:
        partial = _compact_part(
            runtime, part, candidate, keys, slots, active, rows
        )
        if partial is None:
            return None
        accumulated = accumulated.at[rows, keys[None, :]].add(partial)
    else:
        compact = mx.take(accumulated, keys, axis=1)
        for part, candidate in items:
            partial = _compact_part(
                runtime, part, candidate, keys, slots, active, rows
            )
            if partial is None:
                return None
            compact = compact + partial
        # Preserve the position of the dense seed in the addition order.
        # Each inactive slot gets its own padding index so every write has
        # a distinct destination, including when token zero is present.
        width = accumulated.shape[1]
        destinations = mx.where(
            active, keys, width + mx.arange(keys.size, dtype=mx.int64)
        )
        accumulated = mx.put_along_axis(
            mx.pad(accumulated, [(0, 0), (0, keys.size)]),
            destinations[None, :],
            compact,
            axis=1,
        )[:, :width]
        runtime.counts["gpu_shared_indexed_runs"] += 1
        runtime.counts["gpu_shared_indexed_partials"] += len(items)
    runtime.counts["gpu_indexed_union_partials"] += len(items)
    return accumulated.transpose(permutation)


def _merge(runtime, tensor):
    parts, axes = tensor.union, tensor.axes
    if (
        parts is None
        or len(axes) != 2
        or not all(
            len(part.axes) == 2
            and all(
                np.array_equal(a.values, b.values)
                for a, b in zip(part.axes, axes)
            )
            for part in parts
        )
    ):
        return None
    candidates = [_partial(part) for part in parts]
    if not any(candidate is not None for candidate in candidates):
        return None

    shape = tuple(len(a) for a in axes)
    result = mx.zeros(shape, dtype=mx.float32)
    entries = zip(parts, candidates)
    groups = (
        groupby(entries, key=_shared_key)
        if runtime.shared_indexed_union
        else ((None, (item,)) for item in entries)
    )
    for key, items in groups:
        if key is not None:
            result = _add_indexed(runtime, result, tuple(items))
            if result is None:
                return None
            continue
        for part, candidate in items:
            if candidate is not None:
                result = _add_indexed(runtime, result, ((part, candidate),))
                if result is None:
                    return None
                continue
            value = runtime._mx(part.value.get())
            if not mx.issubdtype(value.dtype, mx.floating):
                return None
            data = runtime.broadcast(value, part.value.axes, part.axes, shape)
            valid = runtime._mx(
                _align(part.mask.get(), part.mask.axes, part.axes)
            )
            result = result + mx.where(valid, data, 0)
    return result
