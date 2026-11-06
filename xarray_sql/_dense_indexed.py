"""Accumulate compact indexed partials into an ordered GPU UNION sum."""

from __future__ import annotations

import math

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
    for part, candidate in zip(parts, candidates):
        if candidate is None:
            value = runtime._mx(part.value.get())
            if not mx.issubdtype(value.dtype, mx.floating):
                return None
            data = runtime.broadcast(value, part.value.axes, part.axes, shape)
            valid = runtime._mx(
                _align(part.mask.get(), part.mask.axes, part.axes)
            )
            result = result + mx.where(valid, data, 0)
            continue

        indexed, output, mask = candidate
        key, data = indexed.key.get(), indexed.value.get()
        if (
            not runtime.owns(key)
            or not runtime.owns(data)
            or data.dtype != mx.float32
        ):
            return None
        indexes, _ = runtime.indexes(output[1].values, key)
        keys, slots, active = _coalesce(indexes)
        rows = mx.arange(len(output[0]))[:, None]
        data = runtime.broadcast(
            data,
            indexed.value.axes,
            indexed.local,
            tuple(len(a) for a in indexed.local),
        ).reshape(len(output[0]), -1)
        # Finish each branch's duplicate sum before adding it to the ordered
        # UNION accumulator. Padding contributes zero and has a valid index.
        partial = mx.zeros((len(output[0]), indexes.size), dtype=mx.float32)
        partial = partial.at[rows, slots[None, :]].add(data)
        partial = partial * _mask_at(runtime, mask, output, keys)
        valid = active & _mask_at(runtime, part.mask, output, keys)
        partial = mx.where(valid, partial, 0)
        permutation = tuple(part.axes.index(a) for a in output)
        accumulated = (
            result.transpose(permutation).at[rows, keys[None, :]].add(partial)
        )
        result = accumulated.transpose(permutation)
        runtime.counts["gpu_indexed_union_partials"] += 1
    return result
