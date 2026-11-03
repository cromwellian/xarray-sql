"""Prove dense attention shapes from EinFold's numerical reduction graph.

[EinFold](https://github.com/xqlsystems/einfold) retains SQL row and NULL
masks independently of numerical values. This matcher only accepts masks
computed from structural coordinates; value-dependent masks and probability
dropout retain the original SQL operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class _View:
    node: object
    mapping: tuple = ()

    def axes(self, axes=None):
        mapping = dict(self.mapping)
        return tuple(
            mapping.get(a, a)
            for a in (self.node.axes if axes is None else axes)
        )

    def child(self, node):
        return _view(node, self.mapping)

    def args(self):
        return tuple(self.child(a) for a in self.node.args)

    def same(self, other):
        return self.node is other.node and self.axes() == other.axes()


def _view(node, mapping=()):
    mapping = dict(mapping)
    mapping = {a: mapping.get(a, a) for a in node.axes}
    while node.kind == "rename":
        parent = node.args[0]
        mapping = {a: mapping.get(b, b) for a, b in zip(parent.axes, node.axes)}
        node = parent
    return _View(node, tuple(mapping.items()))


def _bound_same(inner, outer, kept):
    if inner.node is not outer.node:
        return False
    return all(
        a is b
        or (
            a not in kept
            and b not in kept
            and np.array_equal(a.values, b.values)
        )
        for a, b in zip(inner.axes(), outer.axes())
    )


def _bind_reduction(view, reduced):
    """Rename bound reduction coordinates without capturing kept axes."""
    info = view.node.reduction
    original = tuple(a for a in info.row_axes if a not in info.kept)
    if len(original) != len(reduced) or any(
        not np.array_equal(a.values, b.values)
        for a, b in zip(original, reduced)
    ):
        return None
    mapping = dict(view.mapping)
    mapping.update(zip(original, reduced))
    return mapping


def _scalar(value):
    if value.node.kind == "float64":
        return _scalar(value.args()[0])
    if value.node.kind != "array" or value.axes():
        return None
    array = np.asarray(value.node.value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array):
        return None
    return float(array)


def _scale_constant(value):
    scalar = _scalar(value)
    if scalar is None or abs(scalar) > _FLOAT32_MAX:
        return None
    return scalar


def _factors(value):
    if value.node.kind == "*":
        return tuple(f for a in value.args() for f in _factors(a))
    return (value,)


def _static_mask(value, axes):
    if not value.node.host_only or not set(value.axes()).issubset(axes):
        return None
    data = np.asarray(value.node.get())
    ordered = tuple(a for a in axes if a in value.axes())
    data = data.transpose(tuple(value.axes().index(a) for a in ordered))
    return data.reshape(tuple(len(a) if a in ordered else 1 for a in axes))


@dataclass(frozen=True)
class _Attention:
    q: _View
    k: _View
    v: _View
    batch: tuple
    query: object
    key: object
    feature: object
    value_feature: object
    scale: float
    mask: object
    output: tuple
    exponential: _View
    denominator: _View
    gate: object = None


def _probability(exponential, denominator):
    if exponential.node.kind != "exp":
        return None
    summed = denominator.node.reduction
    if summed is None or summed.kind != "sum" or summed.repeated != 1:
        return None
    if not _bound_same(
        denominator.child(summed.value),
        exponential,
        denominator.axes(summed.kept),
    ):
        return None
    difference = exponential.args()[0]
    if difference.node.kind != "-":
        return None
    scores, maximum = difference.args()
    extremum = maximum.node.reduction
    if extremum is None or extremum.kind != "max":
        return None
    if not _bound_same(
        maximum.child(extremum.value), scores, maximum.axes(extremum.kept)
    ):
        return None
    scale = 1.0
    contraction = scores
    if scores.node.kind == "/":
        contraction, divisor = scores.args()
        divisor = _scale_constant(divisor)
        if divisor is None or divisor <= 0:
            return None
        scale = 1.0 / divisor
    elif scores.node.kind == "*":
        left, right = scores.args()
        if _scale_constant(left) is not None:
            scale, contraction = _scale_constant(left), right
        elif _scale_constant(right) is not None:
            scale, contraction = _scale_constant(right), left
    product = contraction.node.reduction
    if product is None or product.kind != "sum" or product.repeated != 1:
        return None
    inputs = _factors(contraction.child(product.value))
    if (
        len(inputs) != 2
        or not np.isfinite(scale)
        or scale <= 0
        or scale > _FLOAT32_MAX
        or np.float32(scale) == 0
    ):
        return None
    return contraction, maximum, denominator, inputs, scale


def _match(tensor):
    """Recognize SUM(exp(scores-max)/SUM(exp(scores-max))*values)."""
    root = _view(tensor)
    reduction = root.node.reduction
    if reduction is None or reduction.kind != "sum" or reduction.repeated != 1:
        return None
    factors = _factors(root.child(reduction.value))
    if len(factors) != 2:
        return None
    for weight, value in (factors, factors[::-1]):
        if weight.node.kind != "/":
            continue
        probability = _probability(*weight.args())
        if probability is None:
            continue
        contraction, maximum, denominator, inputs, scale = probability
        result = _roles(
            root, contraction, maximum, denominator, value, inputs, scale
        )
        if result is not None:
            return result
    return None


def _roles(
    root,
    contraction,
    maximum,
    denominator,
    value,
    inputs,
    scale,
    *,
    output=None,
):
    out = root.axes() if root is not None else output
    score_axes = contraction.axes()
    product = contraction.node.reduction
    for q, k in (inputs, inputs[::-1]):
        query = set(q.axes()) - set(k.axes())
        key = set(k.axes()) - set(q.axes())
        feature = (set(q.axes()) & set(k.axes())) - set(score_axes)
        if len(query) != 1 or len(key) != 1 or len(feature) != 1:
            continue
        query, key, feature = (
            next(iter(query)),
            next(iter(key)),
            next(iter(feature)),
        )
        batch = tuple(a for a in q.axes() if a not in {query, feature})
        value_feature = set(value.axes()) - set(batch) - {key}
        if len(value_feature) != 1:
            continue
        value_feature = next(iter(value_feature))
        if value_feature is query:
            continue
        expected_scores = set(batch) | {query, key}
        groups = set(batch) | {query}
        if (
            len(batch) > 2
            or set(k.axes()) != set(batch) | {key, feature}
            or set(value.axes()) != set(batch) | {key, value_feature}
            or set(out) != set(batch) | {query, value_feature}
            or set(score_axes) != expected_scores
            or set(contraction.axes(product.row_axes))
            != expected_scores | {feature}
            or set(contraction.axes(product.kept)) != expected_scores
            or any(
                len(a) == 0
                for a in (*batch, query, key, feature, value_feature)
            )
        ):
            continue
        masks = []
        valid = True
        reductions = [
            (maximum, expected_scores, groups),
            (denominator, expected_scores, groups),
        ]
        if root is not None:
            reductions.append(
                (root, expected_scores | {value_feature}, set(out))
            )
        for view, rows, kept in reductions:
            info = view.node.reduction
            bound = tuple(a for a in rows if a not in kept)
            mapping = _bind_reduction(view, bound)
            if mapping is None or set(view.axes(info.kept)) != kept:
                valid = False
                break
            masks.append(_view(info.mask, mapping.items()))
        if not valid:
            continue
        masks.append(contraction.child(product.mask))
        axes = (*batch, query, key)
        arrays = [_static_mask(mask, axes) for mask in masks]
        if any(a is None for a in arrays):
            continue
        shape = tuple(len(a) for a in axes)
        mask = np.broadcast_to(arrays[0], shape)
        if not all(
            np.array_equal(mask, np.broadcast_to(a, shape)) for a in arrays[1:]
        ):
            continue
        if not mask.any(axis=-1).all():
            continue
        return _Attention(
            q,
            k,
            value,
            batch,
            query,
            key,
            feature,
            value_feature,
            scale,
            mask,
            out,
            _view(
                denominator.node.reduction.value,
                _bind_reduction(denominator, (key,)).items(),
            ),
            denominator,
        )
    return None
