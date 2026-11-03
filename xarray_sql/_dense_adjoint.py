"""Prove attention adjoints in ddx-generated SQL before kernel selection.

[ddx](https://github.com/xqlsystems/ddx) generates the derivative expressions.
The matcher checks their quotient-rule contributions using a bounded
Laurent-polynomial normal form, including the reduction over valid keys.
This represents products and quotients of symbolic values and reduced sums.
It neither differentiates the SQL nor assumes a benchmark's table names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import _dense_proofs
from ._dense_attention import (
    _bind_reduction,
    _factors,
    _probability,
    _roles,
    _scalar,
    _static_mask,
    _view,
)


@dataclass(frozen=True)
class _Atom:
    name: str
    axes: frozenset
    detail: object = None


def _monomial(*items):
    return frozenset((atom, power) for atom, power in items if power)


def _multiply(left, right):
    if left is None or right is None or len(left) * len(right) > 16:
        return None
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            powers = dict(a)
            for atom, power in b:
                powers[atom] = powers.get(atom, 0) + power
            key = _monomial(*powers.items())
            result[key] = result.get(key, 0.0) + ca * cb
    return {k: v for k, v in result.items() if v}


def _mask(view, attention):
    mask = _dense_proofs._mask(view, attention)
    return None if mask is None or np.any(mask == 2) else mask == 1


def _on_domain(view, attention):
    mask = _dense_proofs._mask(view, attention)
    return mask is not None and np.all(mask[attention.mask] == 1)


class _Polynomial:
    def __init__(self, attention, dp, *, truth=True):
        self.attention = attention
        self.truth = truth
        self.p = _Atom("p", frozenset(attention.exponential.axes()))
        self.z = _Atom("z", frozenset(attention.denominator.axes()))
        self.d = _Atom("d", frozenset(dp.axes()))
        self.symbols = [
            (attention.exponential, self.p),
            (attention.denominator, self.z),
            (dp, self.d),
        ]
        self.visits = 0

    def run(self, view):
        self.visits += 1
        if self.visits > 128:
            return None
        for value, atom in self.symbols:
            if view.same(value):
                if atom is self.d and not self.truth:
                    return {}
                return {_monomial((atom, 1)): 1.0}
        scalar = _scalar(view)
        if scalar is not None:
            return {frozenset(): scalar} if scalar else {}
        kind = view.node.kind
        args = view.args()
        if kind == "float64":
            return self.run(args[0])
        if kind == "where":
            condition = _dense_proofs._mask(
                args[0], self.attention, truth=self.truth
            )
            if condition is None:
                return None
            selected = condition[self.attention.mask]
            if np.all(selected == 1):
                return self.run(args[1])
            if np.all(selected == 0):
                return self.run(args[2])
            return None
        if kind in {"*", "/"}:
            left, right = (self.run(a) for a in args)
            if kind == "/":
                if right is None or len(right) != 1:
                    return None
                powers, coefficient = next(iter(right.items()))
                right = {
                    _monomial(*((a, -p) for a, p in powers)): 1.0 / coefficient
                }
            return _multiply(left, right)
        info = view.node.reduction
        if info is None or info.kind != "sum" or info.repeated != 1:
            return None
        reduced = frozenset(view.axes(info.row_axes)) - frozenset(
            view.axes(info.kept)
        )
        if not self.truth:
            flags = _dense_proofs._mask(
                view.child(info.mask),
                self.attention,
                truth=False,
                forbidden=reduced,
            )
            if flags is not None and np.all(flags[self.attention.mask] == 0):
                return {}
        if not reduced:
            flags = _dense_proofs._mask(
                view.child(info.mask), self.attention, truth=self.truth
            )
            if flags is None:
                return None
            selected = flags[self.attention.mask]
            if np.all(selected == 0):
                return {}
            return (
                self.run(view.child(info.value))
                if np.all(selected == 1)
                else None
            )
        mapping = _bind_reduction(view, (self.attention.key,))
        if mapping is None or frozenset(view.axes(info.kept)) != self.z.axes:
            return None
        reduced = frozenset({self.attention.key})
        mask = _dense_proofs._mask(
            _view(info.mask, mapping.items()), self.attention, truth=self.truth
        )
        if mask is not None and np.all(mask == 0):
            return {}
        if mask is None or not np.array_equal(
            mask, self.attention.mask.astype(np.int8)
        ):
            return None
        inner = self.run(_view(info.value, mapping.items()))
        if inner is None:
            return None
        result = {}
        for monomial, coefficient in inner.items():
            inside = _monomial(
                *((a, p) for a, p in monomial if a.axes & reduced)
            )
            outside = _monomial(
                *((a, p) for a, p in monomial if not a.axes & reduced)
            )
            if not inside:
                return None
            summed = _Atom("sum", self.z.axes, (reduced, inside))
            term = outside | _monomial((summed, 1))
            result[term] = result.get(term, 0.0) + coefficient
        return result

    def expected(self):
        summed = _Atom(
            "sum",
            self.z.axes,
            (
                frozenset({self.attention.key}),
                _monomial((self.p, 1), (self.d, 1)),
            ),
        )
        return {
            _monomial(
                (self.p, 1), (self.z, -1), (self.d, 1)
            ): self.attention.scale,
            _monomial(
                (self.p, 1), (self.z, -2), (summed, 1)
            ): -self.attention.scale,
        }


def _candidates(root):
    pending, seen = [root], set()
    while pending and len(seen) < 128:
        view = pending.pop()
        key = view.node, view.axes()
        if key in seen:
            continue
        seen.add(key)
        yield view
        info = view.node.reduction
        if view.node.kind == "exp":
            continue
        if info is not None:
            if not set(view.axes(info.row_axes)) - set(view.axes(info.kept)):
                pending.append(view.child(info.value))
        elif view.node.kind == "where":
            pending.extend(view.args()[1:])
        else:
            pending.extend(view.args())


def _parts(root):
    if root.node.union is None or len(root.node.union) != 2:
        return None
    result = []
    for part in root.node.union:
        if len(part.axes) != len(root.axes()) or not all(
            np.array_equal(a.values, b.values)
            for a, b in zip(part.axes, root.axes())
        ):
            return None
        mapping = dict(root.mapping)
        mapping.update(zip(part.axes, root.axes()))
        result.append(
            (
                _view(part.value, mapping.items()),
                _view(part.mask, mapping.items()),
            )
        )
    return result


def _score_adjoint(root):
    parts = _parts(root)
    if parts is None:
        return None
    nodes = [node for part, _ in parts for node in _candidates(part)]
    for p in (n for n in nodes if n.node.kind == "exp"):
        for z in (
            n
            for n in nodes
            if n.node.reduction is not None and n.node.reduction.kind == "sum"
        ):
            probability = _probability(p, z)
            if probability is None:
                continue
            contraction, maximum, denominator, inputs, scale = probability
            for dp in nodes:
                info = dp.node.reduction
                if (
                    info is None
                    or info.kind != "sum"
                    or info.repeated != 1
                    or set(dp.axes(info.kept)) != set(p.axes())
                ):
                    continue
                factors = _factors(dp.child(info.value))
                if len(factors) != 2:
                    continue
                for dy, v in (factors, factors[::-1]):
                    match = _roles(
                        None,
                        contraction,
                        maximum,
                        denominator,
                        v,
                        inputs,
                        scale,
                        output=dy.axes(),
                    )
                    if (
                        match is None
                        or len(match.query) < 32
                        or set(dp.axes(info.row_axes))
                        != set(p.axes()) | {match.value_feature}
                    ):
                        continue
                    gate = dp.child(info.mask)
                    if not set(gate.axes()).issubset(
                        (*match.batch, match.query)
                    ):
                        continue
                    static = _static_mask(gate, (*match.batch, match.query))
                    if static is None or not static.all():
                        match = replace(match, gate=gate)
                    if not _on_domain(gate, match):
                        continue
                    if not all(_on_domain(mask, match) for _, mask in parts):
                        continue
                    normal = _Polynomial(match, dp)
                    left, right = (normal.run(value) for value, _ in parts)
                    if left is None or right is None:
                        continue
                    combined = dict(left)
                    for term, c in right.items():
                        combined[term] = combined.get(term, 0.0) + c
                    expected = normal.expected()
                    if combined.keys() == expected.keys() and all(
                        np.isclose(combined[k], v, rtol=1e-12, atol=0)
                        for k, v in expected.items()
                    ):
                        if match.gate is not None:
                            zero = _Polynomial(match, dp, truth=False)
                            if any(zero.run(value) != {} for value, _ in parts):
                                continue
                        return match, dy
    return None


@dataclass(frozen=True)
class _Gradient:
    attention: object
    cotangent: object
    output: tuple
    argument: int
    source: tuple


def _match(tensor):
    root = _view(tensor)
    info = root.node.reduction
    if info is None or info.kind != "sum" or info.repeated != 1:
        return None
    factors = _factors(root.child(info.value))
    if len(factors) != 2:
        return None
    for score, other in (factors, factors[::-1]):
        unions = [n for n in _candidates(score) if n.node.union is not None]
        if len(unions) != 1:
            continue
        union = unions[0]
        result = _score_adjoint(union)
        if result is None:
            continue
        attention, dy = result
        identity = _Polynomial(attention, union)
        symbol = _Atom("score", frozenset(union.axes()))
        identity.symbols = [(union, symbol)]
        if identity.run(score) != {_monomial((symbol, 1)): 1.0}:
            continue
        if other.node is attention.k.node:
            argument, operand, sequence = 0, attention.k, attention.query
            shared = (*attention.batch, attention.key)
        elif other.node is attention.q.node:
            argument, operand, sequence = 1, attention.q, attention.key
            shared = (*attention.batch, attention.query)
        else:
            continue
        mapping = dict(zip(operand.axes(), other.axes()))
        if any(mapping[a] is not a for a in shared):
            continue
        feature = mapping[attention.feature]
        if feature in {*attention.batch, attention.query, attention.key}:
            continue
        output = (*attention.batch, sequence, feature)
        mask = _mask(root.child(info.mask), attention)
        if (
            set(root.axes()) == set(output)
            and set(root.axes(info.kept)) == set(output)
            and set(root.axes(info.row_axes))
            == {
                *attention.batch,
                attention.query,
                attention.key,
                feature,
            }
            and mask is not None
            and np.array_equal(mask, attention.mask)
        ):
            return _Gradient(attention, dy, root.axes(), argument, output)
    return None
