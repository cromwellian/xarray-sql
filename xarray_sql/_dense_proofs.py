"""Structural Boolean proofs for SQL row-presence expressions.

Unknown numerical predicates remain unknown. A row gate can be assumed
true/false only outside reductions that bind one of its axes. No GPU
array is read to establish these implications.
The 0/1/2 results mean false/true/unknown proof states, not SQL NULLs.
"""

from __future__ import annotations

import numpy as np

from ._dense_attention import _scalar, _view


def _align(data, source, target):
    ordered = tuple(a for a in target if a in source)
    return data.transpose(tuple(source.index(a) for a in ordered)).reshape(
        tuple(len(a) if a in source else 1 for a in target)
    )


def _conjuncts(view, depth=0):
    if depth > 32:
        return (view,)
    if view.node.kind in {"bool", "float64"}:
        return _conjuncts(view.args()[0], depth + 1)
    if view.node.kind == "and":
        return tuple(x for a in view.args() for x in _conjuncts(a, depth + 1))
    if view.node.kind == ">" and _scalar(view.args()[1]) == 0:
        count = view.args()[0]
        info = count.node.reduction
        if (
            info is not None
            and info.kind == "sum"
            and _scalar(count.child(info.value)) == 1
            and np.asarray(info.value.value).dtype.kind in "bi"
            and count.node.count_bound is not None
            and count.node.count_bound < 1 << 63
        ):
            kept = frozenset(count.axes(info.kept))
            reduced = frozenset(count.axes(info.row_axes)) - kept
            if all(len(a) > 0 for a in reduced):
                masks = _conjuncts(count.child(info.mask), depth + 1)
                retained = tuple(
                    m for m in masks if set(m.axes()).issubset(kept)
                )
                structural = tuple(
                    m for m in masks if not set(m.axes()).issubset(kept)
                )
                if all(m.node.host_only for m in structural):
                    axes = tuple(
                        dict.fromkeys(a for m in structural for a in m.axes())
                    )
                    data = np.asarray(True)
                    for m in structural:
                        data = data & _align(
                            np.asarray(m.node.get()), m.axes(), axes
                        )
                    present = np.any(
                        data,
                        axis=tuple(
                            i for i, a in enumerate(axes) if a not in kept
                        ),
                    )
                    if np.all(present):
                        return retained
    return (view,)


class _BooleanProof:
    def __init__(self, gate=None, truth=True):
        self.gate, self.truth = gate, truth
        self.remaining = 512
        self.memo = {}
        self.assumptions = (
            set()
            if gate is None
            else {(v.node, v.axes()) for v in _conjuncts(gate)}
        )

    def run(self, view, forbidden=frozenset()):
        key = view.node, view.axes(), forbidden
        if key not in self.memo:
            self.memo[key] = self._visit(view, forbidden)
        return self.memo[key]

    def _visit(self, view, forbidden):
        self.remaining -= 1
        if self.remaining < 0:
            return (), np.asarray(2, dtype=np.int8)
        if self.gate is not None and not set(self.gate.axes()) & forbidden:
            terms = {(v.node, v.axes()) for v in _conjuncts(view)}
            if self.truth and (view.node, view.axes()) in self.assumptions:
                return (), np.asarray(1, dtype=np.int8)
            if not self.truth and self.assumptions.issubset(terms):
                return (), np.asarray(0, dtype=np.int8)
            if view.same(self.gate):
                return (), np.asarray(int(self.truth), dtype=np.int8)
        if view.node.host_only:
            return view.axes(), np.asarray(view.node.get(), dtype=np.int8)
        kind = view.node.kind
        args = view.args()
        if kind in {"and", "or", "where"}:
            children = [self.run(a, forbidden) for a in args]
            axes = tuple(
                dict.fromkeys(a for group, _ in children for a in group)
            )
            arrays = [_align(v, local, axes) for local, v in children]
            if kind == "and":
                a, b = arrays
                return axes, np.where(
                    (a == 0) | (b == 0), 0, np.where((a == 1) & (b == 1), 1, 2)
                )
            if kind == "or":
                a, b = arrays
                return axes, np.where(
                    (a == 1) | (b == 1), 1, np.where((a == 0) & (b == 0), 0, 2)
                )
            condition, yes, no = arrays
            return axes, np.where(
                condition == 1,
                yes,
                np.where(condition == 0, no, np.where(yes == no, yes, 2)),
            )
        if kind in {"not", "bool"}:
            axes, data = self.run(args[0], forbidden)
            return axes, np.where(
                data == 2, 2, 1 - data
            ) if kind == "not" else data
        if kind == ">" and _scalar(args[1]) == 0:
            value = args[0]
            info = value.node.reduction
            if (
                info is not None
                and info.kind == "sum"
                and _scalar(value.child(info.value)) == 1
                and np.asarray(info.value.value).dtype.kind in "bi"
                and value.node.count_bound is not None
                and value.node.count_bound < 1 << 63
            ):
                reduced = frozenset(value.axes(info.row_axes)) - frozenset(
                    value.axes(info.kept)
                )
                if any(len(a) == 0 for a in reduced):
                    return (), np.asarray(0, dtype=np.int8)
                axes, data = self.run(
                    value.child(info.mask), forbidden | reduced
                )
                summed = tuple(i for i, a in enumerate(axes) if a in reduced)
                result = np.where(
                    np.any(data == 1, axis=summed),
                    1,
                    np.where(np.any(data == 2, axis=summed), 2, 0),
                )
                return tuple(a for a in axes if a not in reduced), result
        parts = view.node.presence
        if parts is not None:
            result = None
            for part in parts:
                if (
                    len(part.axes) != len(view.axes())
                    or len(set(part.axes)) != len(part.axes)
                    or any(
                        not np.array_equal(a.values, b.values)
                        for a, b in zip(part.axes, view.axes())
                    )
                ):
                    return (), np.asarray(2, dtype=np.int8)
                mapping = dict(zip(part.axes, view.axes()))
                axes, data = self.run(
                    _view(part.mask, mapping.items()), forbidden
                )
                if not set(axes).issubset(view.axes()):
                    return (), np.asarray(2, dtype=np.int8)
                data = np.broadcast_to(
                    _align(data, axes, view.axes()),
                    tuple(len(a) for a in view.axes()),
                )
                result = (
                    data
                    if result is None
                    else np.where(
                        (result == 1) | (data == 1),
                        1,
                        np.where((result == 0) & (data == 0), 0, 2),
                    )
                )
            if result is not None:
                return view.axes(), result
        return (), np.asarray(2, dtype=np.int8)


def _mask(view, attention, *, truth=True, forbidden=frozenset()):
    gate = getattr(attention, "gate", None)
    axes, data = _BooleanProof(gate, truth).run(view, forbidden)
    target = (*attention.batch, attention.query, attention.key)
    if not set(axes).issubset(target):
        return None
    return np.broadcast_to(_align(data, axes, target), attention.mask.shape)
