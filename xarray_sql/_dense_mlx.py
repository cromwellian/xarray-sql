"""MLX array operations for EinFold's compact numerical programs.

[MLX](https://github.com/ml-explore/mlx) evaluates the arithmetic chosen by
[EinFold](https://github.com/xqlsystems/einfold). Compiled functions take
every numerical leaf as an argument, so a cached function never captures
parameter values from an earlier training step. Coordinates stay on the CPU.
"""

from __future__ import annotations

import os
import string
from collections import Counter
from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

import mlx.core as mx
import numpy as np
from einfold.dense_program import (
    _CACHE,
    _MAX_BYTES,
    _OPS,
    _align,
    _Decline,
    _op,
    _Tensor,
)

from . import (
    _dense_adjoint,
    _dense_attention,
    _dense_indexed,
    _dense_proofs,
    _dense_segment,
)


@dataclass
class _AttentionVJP:
    attention: object
    cotangent: object
    gradients: list

    @property
    def nbytes(self):
        return sum(g.nbytes for g in self.gradients)


def _remainder(a, b):
    result = mx.remainder(a, b)
    return mx.where((result != 0) & ((a < 0) != (b < 0)), result - b, result)


_FUNCTIONS = {
    "+": mx.add,
    "-": mx.subtract,
    "*": mx.multiply,
    "/": mx.divide,
    "%": _remainder,
    "=": mx.equal,
    "!=": mx.not_equal,
    "<": mx.less,
    "<=": mx.less_equal,
    ">": mx.greater,
    ">=": mx.greater_equal,
    "and": mx.logical_and,
    "or": mx.logical_or,
    "not": mx.logical_not,
    "neg": mx.negative,
    "where": mx.where,
    "abs": mx.abs,
    "sqrt": mx.sqrt,
    "exp": mx.exp,
    "erf": mx.erf,
    "ln": mx.log,
    "log": mx.log10,
    "tanh": mx.tanh,
    "isnan": mx.isnan,
    "iszero": lambda x: x == 0,
    "signum": mx.sign,
    "sign": mx.sign,
    "sin": mx.sin,
    "cos": mx.cos,
    "power": mx.power,
    "pow": mx.power,
    "floor": mx.floor,
    "ceil": mx.ceil,
    "minimum": mx.minimum,
    "maximum": mx.maximum,
    "float64": lambda x: x.astype(mx.float32),
    "int64": lambda x: x.astype(mx.int64),
    "bool": lambda x: x.astype(mx.bool_),
}


class _Runtime:
    def __init__(self):
        if os.environ.get("MLX_ENABLE_TF32") != "0":
            raise ValueError("launch Python with MLX_ENABLE_TF32=0")
        self.compiled = {}
        self.counts = Counter()
        self.tracing = False
        self.direct_union = os.environ.get("XQL_MLX_DIRECT_UNION", "1") != "0"
        self.indexed_union = os.environ.get("XQL_MLX_INDEXED_UNION") == "1"
        self.shared_indexed_union = (
            self.indexed_union
            and os.environ.get("XQL_MLX_SHARED_INDEXED_UNION") == "1"
        )
        self._lookup_indexes = {}
        self.fixed_segments = os.environ.get("XQL_MLX_FIXED_SEGMENTS") == "1"
        attention = os.environ.get("XQL_MLX_ATTENTION", "1")
        if attention not in {"0", "off", "forward", "1", "all"}:
            raise ValueError("XQL_MLX_ATTENTION must be off, forward or all")
        self.fast_attention = attention not in {"0", "off"}
        self.fast_attention_backward = attention in {"1", "all"}
        self.attention_cache = WeakKeyDictionary()

    def owns(self, array):
        return isinstance(array, mx.array)

    def boolean(self, value):
        self.counts["gpu_status_readbacks"] += 1
        return bool(value.item())

    def status(self, values, flags):
        status = mx.array(True)
        for flag in flags:
            status = mx.logical_and(status, mx.all(self._mx(flag)))
        for value in values:
            status = mx.logical_and(
                status, mx.all(mx.isfinite(self._mx(value)))
            )
        return status

    def validate(self, values, flags):
        status = self.status(values, flags)
        mx.eval(values, status)
        if not self.boolean(status):
            raise ValueError(
                "training update must be rectangular, nonnullable and finite"
            )

    def compile_training(self, function):
        self.counts["gpu_training_compilations"] += 1
        return mx.compile(function)

    def finish_training(self, arrays, status):
        self.counts["gpu_training_executions"] += 1
        if status is None:
            mx.eval(arrays)
            return
        mx.eval(arrays, status)
        if not self.boolean(status):
            raise ValueError(
                "training update must be rectangular, nonnullable and finite"
            )

    def indices(self, values):
        if values.size and (
            int(values.min()) < -(1 << 63) or int(values.max()) >= (1 << 63)
        ):
            raise _Decline("device index is outside exact int64 range")
        self.counts["gpu_input_arrays"] += 1
        self.counts["gpu_input_bytes"] += values.nbytes
        return mx.array(values, dtype=mx.int64)

    def random_key(self, seed, step):
        return mx.bitwise_xor(
            mx.random.key(seed), mx.array([0, step], dtype=mx.uint32)
        )

    def dropout(self, key, shapes, probability):
        keys = mx.random.split(key, len(shapes))
        return [
            mx.random.bernoulli(1 - probability, shape=shape, key=k).astype(
                mx.float32
            )
            / (1 - probability)
            for k, shape in zip(keys, shapes)
        ]

    def memory(self):
        return {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        }

    def copy(self, value):
        return mx.array(value)

    def numpy(self, value):
        if self.owns(value):
            if self.tracing:
                raise _Decline(
                    "compiled numerical graph requires a host conversion"
                )
            self.counts["gpu_readbacks"] += 1
            self.counts["gpu_readback_bytes"] += value.size * value.itemsize
        return np.asarray(value)

    def array(self, value):
        array = np.asarray(value)
        if array.dtype.kind == "f":
            # ddx uses the extreme float64 value as a masked MAX sentinel.
            # It stays beyond every supported finite float32 input. An
            # observable infinity is rejected at the final SQL boundary.
            if array.ndim == 0 and abs(array) == np.finfo(np.float64).max:
                return mx.array(float("-inf") if array < 0 else float("inf"))
            finite = np.isfinite(array)
            if (array.ndim and not finite.all()) or np.any(
                np.abs(array[finite]) > np.finfo(np.float32).max
            ):
                raise _Decline(
                    f"GPU input outside finite float32 range: {array.dtype}, shape={array.shape}, finite={finite.sum()}/{array.size}"
                )
            self.counts["gpu_input_arrays"] += 1
            self.counts["gpu_input_bytes"] += array.nbytes
            return mx.array(array, dtype=mx.float32)
        return array

    def _mx(self, value):
        return value if self.owns(value) else mx.array(value)

    def pointwise(self, tensor):
        leaves, instructions, seen = [], [], {}

        def describe(node):
            if node in seen:
                return seen[node]
            if node.kind in _OPS and node.cached is None and len(seen) < 64:
                children = tuple(describe(a) for a in node.args)
                instruction = node.kind, children
            else:
                leaves.append(_align(node.get(), node.axes, tensor.axes))
                instruction = "leaf", len(leaves) - 1
            at = len(instructions)
            seen[node] = at
            instructions.append(instruction)
            return at

        describe(tensor)
        signature = tuple(instructions)
        if not any(self.owns(a) for a in leaves):
            return None
        if signature not in self.compiled:

            def execute(*args):
                values = []
                for kind, children in signature:
                    if kind == "leaf":
                        values.append(args[children])
                    else:
                        values.append(
                            _FUNCTIONS[kind](*(values[c] for c in children))
                        )
                return values[-1]

            self.compiled[signature] = mx.compile(execute)
            self.counts["gpu_fused_programs"] += 1
        self.counts["gpu_fused_executions"] += 1
        return self.compiled[signature](*(self._mx(a) for a in leaves))

    def reduction(self, tensor):
        if self.tracing and self.direct_union and self.indexed_union:
            result = _dense_indexed._merge(self, tensor)
            if result is not None:
                return result
        if not self.fast_attention:
            return None
        if self.fast_attention_backward:
            gradient = _dense_adjoint._match(tensor)
            if gradient is not None:
                result = self._attention_gradient(gradient)
                if result is not None:
                    return result
            result = self._attention_value_gradient(tensor)
            if result is not None:
                return result
        match = _dense_attention._match(tensor)
        if match is None:
            return None
        prepared = self._attention_inputs(match)
        if prepared is None:
            return None
        arrays, mask = prepared
        try:
            result = mx.fast.scaled_dot_product_attention(
                *arrays, scale=match.scale, mask=mask, force_fused=True
            )
        except ValueError:
            return None
        batch = tuple(len(a) for a in match.batch)
        result = result.reshape(
            (*batch, len(match.query), len(match.value_feature))
        )
        source = (*match.batch, match.query, match.value_feature)
        self.counts["gpu_attention_forwards"] += 1
        return result.transpose(tuple(source.index(a) for a in match.output))

    def _attention_inputs(self, match):
        batch = tuple(len(a) for a in match.batch)
        leading = (*batch, 1) if len(batch) == 1 else batch or (1, 1)
        arrays = []
        for view, axes in [
            (match.q, (*match.batch, match.query, match.feature)),
            (match.k, (*match.batch, match.key, match.feature)),
            (match.v, (*match.batch, match.key, match.value_feature)),
        ]:
            array = view.node.get()
            if not self.owns(array) or array.dtype != mx.float32:
                return None
            array = array.transpose(tuple(view.axes().index(a) for a in axes))
            arrays.append(array.reshape((*leading, *array.shape[-2:])))
        mask = match.mask
        if mask.all():
            mask = None
        elif len(match.query) == len(match.key) and np.array_equal(
            mask,
            np.broadcast_to(
                np.arange(len(match.query))[:, None]
                >= np.arange(len(match.key))[None, :],
                mask.shape,
            ),
        ):
            mask = "causal"
        else:
            mask = mx.array(
                mask.reshape((*leading, len(match.query), len(match.key)))
            )
        return arrays, mask

    def _attention_gradient(self, gradient):
        cache = _CACHE.get()
        if cache is None or os.environ.get("MLX_SDPA_VJP_FALLBACK", "0") != "0":
            return None
        match, dy = gradient.attention, gradient.cotangent
        layouts = [
            (match.q, (*match.batch, match.query, match.feature)),
            (match.k, (*match.batch, match.key, match.feature)),
            (match.v, (*match.batch, match.key, match.value_feature)),
            (dy, (*match.batch, match.query, match.value_feature)),
        ]
        key = (
            match.exponential.node,
            match.denominator.node,
            None
            if match.gate is None
            else (
                match.gate.node,
                tuple(
                    match.gate.axes().index(a)
                    if a in match.gate.axes()
                    else None
                    for a in (*match.batch, match.query)
                ),
            ),
            tuple(
                (v.node, tuple(v.axes().index(a) for a in axes))
                for v, axes in layouts
            ),
        )
        entries = self.attention_cache.setdefault(cache, {})
        slot = entries.get(key)
        record = None if slot is None else slot.cached
        if record is None:
            prepared = self._attention_inputs(match)
            if prepared is None:
                return None
            arrays, mask = prepared
            if any(a.size * 8 > _MAX_BYTES.get() for a in arrays):
                return None
            data = dy.node.get()
            if not self.owns(data) or data.dtype != mx.float32:
                return None
            axes = layouts[-1][1]
            data = data.transpose(
                tuple(dy.axes().index(a) for a in axes)
            ).reshape((*arrays[0].shape[:-1], arrays[2].shape[-1]))
            if match.gate is not None:
                gate = self.broadcast(
                    match.gate.node.get(),
                    match.gate.axes(),
                    axes,
                    tuple(len(a) for a in axes),
                )
                data = mx.where(gate.reshape(data.shape), data, 0)

            def primitive(q, k, v):
                return mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=match.scale, mask=mask, force_fused=True
                )

            try:
                _, gradients = mx.vjp(primitive, arrays, [data])
            except ValueError:
                return None
            # Share the sibling kernel outputs within the same byte-bounded
            # LRU as ordinary tensors. Eviction clears this private carrier;
            # the side table then contains metadata, not extra array owners.
            slot = _Tensor("reduce")
            slot.cached = _AttentionVJP(match, dy, gradients)
            record = cache.remember(slot)
            entries[key] = slot
            self.counts["gpu_attention_vjps"] += 1
        batch = tuple(len(a) for a in match.batch)
        sequence = match.query if gradient.argument == 0 else match.key
        if slot in cache.entries:
            cache.entries.move_to_end(slot)
        result = record.gradients[gradient.argument].reshape(
            (*batch, len(sequence), len(match.feature))
        )
        self.counts["gpu_attention_gradient_outputs"] += 1
        return result.transpose(
            tuple(gradient.source.index(a) for a in gradient.output)
        )

    def _attention_value_gradient(self, tensor):
        cache = _CACHE.get()
        entries = (
            self.attention_cache.get(cache, {}) if cache is not None else {}
        )
        root = _dense_attention._view(tensor)
        info = root.node.reduction
        if (
            not entries
            or info is None
            or info.kind != "sum"
            or info.repeated != 1
        ):
            return None
        factors = _dense_attention._factors(root.child(info.value))
        if len(factors) != 2:
            return None
        for weight, dy in (factors, factors[::-1]):
            if weight.node.kind != "/":
                continue
            p, z = weight.args()
            for slot in entries.values():
                record = slot.cached
                if record is None:
                    continue
                match = record.attention
                if (
                    p.node is not match.exponential.node
                    or z.node is not match.denominator.node
                    or dy.node is not record.cotangent.node
                ):
                    continue
                mapping = dict(zip(match.exponential.axes(), p.axes()))
                renamed_dy = dict(zip(record.cotangent.axes(), dy.axes()))
                if any(
                    renamed_dy[a] is not mapping[a]
                    for a in (*match.batch, match.query)
                ):
                    continue
                batch = tuple(mapping[a] for a in match.batch)
                query, key = mapping[match.query], mapping[match.key]
                feature = renamed_dy[match.value_feature]
                if feature in {*batch, query, key}:
                    continue
                output = (*batch, key, feature)
                if (
                    set(root.axes()) != set(output)
                    or set(root.axes(info.kept)) != set(output)
                    or set(root.axes(info.row_axes))
                    != {*batch, query, key, feature}
                ):
                    continue
                gate = (
                    None
                    if match.gate is None
                    else _dense_attention._view(
                        match.gate.node,
                        (
                            (source, renamed_dy[target])
                            for source, target in match.gate.mapping
                        ),
                    )
                )
                context = replace(
                    match, batch=batch, query=query, key=key, gate=gate
                )
                mask = _dense_adjoint._mask(root.child(info.mask), context)
                if mask is None or not np.array_equal(mask, match.mask):
                    continue
                if gate is not None:
                    absent = _dense_proofs._mask(
                        root.child(info.mask), context, truth=False
                    )
                    if absent is None or np.any(absent[match.mask] != 0):
                        zero = _dense_adjoint._Polynomial(
                            context, dy, truth=False
                        )
                        zero.symbols = []
                        if zero.run(dy) != {}:
                            continue
                result = record.gradients[2].reshape(
                    (*tuple(len(a) for a in batch), len(key), len(feature))
                )
                self.counts["gpu_attention_gradient_outputs"] += 1
                return result.transpose(
                    tuple(output.index(a) for a in root.axes())
                )
        return None

    def take(self, data, indexes, axis):
        return (
            mx.take(self._mx(data), self._mx(indexes), axis=axis)
            if self.owns(data) or self.owns(indexes)
            else np.take(data, indexes, axis=axis)
        )

    def indexes(self, domain, values):
        if (
            self.tracing
            and self.shared_indexed_union
            and values.ndim
            and values.size < len(domain)
        ):
            # Cache only within one trace. A join alias can rename the axes
            # while still passing the same ordered device values.
            entries = self._lookup_indexes.setdefault(id(values), [])
            for source, coordinates, result in entries:
                if (
                    source is values
                    and coordinates.dtype == domain.dtype
                    and np.array_equal(coordinates, domain)
                ):
                    return result
            result = self._compute_indexes(domain, values)
            entries.append((values, domain, result))
            return result
        return self._compute_indexes(domain, values)

    def _compute_indexes(self, domain, values):
        if domain.dtype.kind not in "iu" or not len(domain):
            raise _Decline("GPU index domain must contain integers")
        if int(domain.min()) < -(1 << 63) or int(domain.max()) >= (1 << 63):
            raise _Decline("GPU index domain is outside exact int64 range")
        domain = domain.astype(np.int64, copy=False)
        # A proven contiguous domain maps integer labels directly to offsets.
        # Clip before subtraction to avoid int64 overflow for invalid labels;
        # validity still reflects the original values.
        if (
            os.environ.get("XQL_MLX_DIRECT_INDEX", "1") != "0"
            and values.dtype == mx.int64
            and int(domain[-1]) - int(domain[0]) == len(domain) - 1
            and np.all(domain[1:] > domain[:-1])
        ):
            lo = mx.array(int(domain[0]), dtype=mx.int64)
            hi = mx.array(int(domain[-1]), dtype=mx.int64)
            return mx.clip(values, lo, hi) - lo, (values >= lo) & (values <= hi)
        order = np.argsort(domain)
        sorted_domain = mx.array(domain[order])
        at = mx.minimum(mx.searchsorted(sorted_domain, values), len(domain) - 1)
        indexes = mx.take(mx.array(order), at)
        return indexes, mx.take(mx.array(domain), indexes) == values

    def index(self, data, indexes):
        return (
            self._mx(data)[tuple(self._mx(i) for i in indexes)]
            if self.owns(data) or any(self.owns(i) for i in indexes)
            else data[tuple(indexes)]
        )

    def contract(self, arrays, labels, output):
        converted = [self._mx(a) for a in arrays]
        if not any(mx.issubdtype(a.dtype, mx.floating) for a in converted):
            for array in arrays:
                if self.owns(array):
                    if array.dtype != mx.bool_:
                        raise _Decline(
                            "GPU integer contraction requires a count mask"
                        )
                elif np.any((array < -1) | (array > 1)):
                    raise _Decline(
                        "GPU integer contraction requires unit factors"
                    )
            dimensions = {
                label: size
                for a, names in zip(converted, labels)
                for label, size in zip(names, a.shape)
            }
            used = tuple(sorted(dimensions))
            value = mx.array(1, dtype=mx.int64)
            for a, names in zip(converted, labels):
                ordered = [n for n in used if n in names]
                a = a.transpose(tuple(names.index(n) for n in ordered))
                a = a.reshape(
                    tuple(dimensions[n] if n in names else 1 for n in used)
                )
                value = value * a.astype(mx.int64)
            value = mx.sum(
                value,
                axis=tuple(i for i, n in enumerate(used) if n not in output),
            )
            remaining = [n for n in used if n in output]
            return value.transpose(tuple(remaining.index(n) for n in output))
        equation = (
            ",".join(
                "".join(string.ascii_letters[i] for i in label)
                for label in labels
            )
            + "->"
            + "".join(string.ascii_letters[i] for i in output)
        )
        # Mask/count operands are promoted before contraction; a boolean
        # reduction otherwise counts presence rather than SQL rows.
        arrays = [a.astype(mx.float32) for a in converted]
        self.counts["gpu_contractions"] += 1
        return mx.einsum(equation, *arrays)

    def broadcast(self, data, source, target, shape):
        return mx.broadcast_to(_align(self._mx(data), source, target), shape)

    def extreme(self, kind, data, source, mask, mask_axes, used, kept):
        shape = tuple(len(a) for a in used)
        if any(size == 0 for size in shape):
            raise _Decline("empty GPU extremum")
        data = self.broadcast(data, source, used, shape)
        mask = self.broadcast(mask, mask_axes, used, shape)
        data = mx.where(
            mask, data, -float("inf") if kind == "max" else float("inf")
        )
        axes = tuple(i for i, a in enumerate(used) if a not in kept)
        return (mx.max if kind == "max" else mx.min)(data, axis=axes)

    def scatter(self, data, indexes, size, *, count_bound):
        data = self._mx(data)
        integer = data.dtype == mx.int64
        if integer:
            if count_bound > np.iinfo(np.int32).max:
                raise _Decline("GPU integer scatter exceeds exact int32 range")
            data = data.astype(mx.int32)
        self.counts["gpu_scatter_reductions"] += 1
        if (
            self.fixed_segments
            and self.owns(indexes)
            and data.dtype == mx.float32
        ):
            self.counts["gpu_fixed_segment_reductions"] += 1
            return _dense_segment._sum(data, indexes, size)
        result = mx.zeros((data.shape[0], size), dtype=data.dtype)
        result = result.at[
            mx.arange(data.shape[0])[:, None], mx.array(indexes)
        ].add(data)
        return result.astype(mx.int64) if integer else result

    def merge(self, parts, keys, key, axes, *, flags=None):
        result = mx.zeros(tuple(len(a) for a in axes), dtype=mx.float32)
        for part in parts:
            source = tuple(part.columns[k].tensor.axis for k in keys)
            shape = tuple(len(a) for a in source)
            if flags is not None:
                mask = (
                    part.present
                    if flags == "rows"
                    else _op("and", part.present, part.columns[key].valid)
                )
                data = mx.ones(shape, dtype=mx.float32)
            else:
                col = part.columns[key]
                value = col.tensor.get()
                if self.owns(value) and not mx.issubdtype(
                    value.dtype, mx.floating
                ):
                    raise _Decline(
                        "GPU integer partial aggregate requires relational execution"
                    )
                mask = _op("and", part.present, col.valid)
                data = self.broadcast(value, col.tensor.axes, source, shape)
            valid = self.broadcast(mask.get(), mask.axes, source, shape)
            data = mx.where(valid, data, 0)
            if (
                axes
                and self.direct_union
                and len(axes) == len(source)
                and all(
                    np.array_equal(a.values, s.values)
                    for a, s in zip(axes, source)
                )
            ):
                # A grouped branch has one value per coordinate. Matching
                # ordered domains make its scatter the identity mapping.
                result = result + data
                self.counts["gpu_union_direct_adds"] += 1
            elif axes:
                indexes = [
                    np.searchsorted(a.values, s.values)
                    for a, s in zip(axes, source)
                ]
                grid = [
                    mx.array(
                        i.reshape(
                            tuple(
                                len(i) if j == at else 1
                                for j in range(len(axes))
                            )
                        )
                    )
                    for at, i in enumerate(indexes)
                ]
                result = result.at[tuple(grid)].add(data)
                self.counts["gpu_union_scatter_adds"] += 1
            else:
                result = result + data
        self.counts["gpu_union_reductions"] += 1
        return result
