"""Capture a compiled Modded-NanoGPT update and test two contraction policies.

The graph report describes logical values and their uses in one topological
order. MLX may alias, copy, or donate the underlying buffers. Use allocator
samples or Instruments for device measurements. Probe policies are confined
to this benchmark process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
import string
import sys
import time
from collections import Counter, defaultdict
from graphlib import TopologicalSorter
from pathlib import Path

import mlx.core as mx
import numpy as np

from xarray_sql import _dense_indexed
from xarray_sql._dense_mlx import _Runtime

_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _ROOT / "benchmarks/ml/04_modded_nanogpt.py"
sys.path.insert(0, str(_DRIVER.parent))
from _modded_sql import _SQLTrainer


def _graph_summary(path):
    kinds, producer, inputs, sinks = {}, {}, defaultdict(list), set()
    operation = re.compile(r'^\{ (\d+) \[label ="(.*)", shape=rectangle\]; \}$')
    edge = re.compile(r'^("?\w+"?) -> ("?\w+"?)$')
    sink = re.compile(r'^\{ rank=sink; ("\w+"); \}$')
    for line in path.read_text().splitlines():
        if match := operation.fullmatch(line):
            node, kind = match.groups()
            kinds[node] = kind
        elif match := edge.fullmatch(line):
            source, target = match.groups()
            if source.startswith('"'):
                inputs[target].append(source)
            else:
                producer[target] = source
        elif match := sink.fullmatch(line):
            sinks.add(match[1])
    if not kinds or not sinks:
        raise ValueError("unrecognized MLX DOT graph")

    order = list(
        TopologicalSorter(
            {
                node: {producer[v] for v in inputs[node] if v in producer}
                for node in kinds
            }
        ).static_order()
    )
    position = {node: i + 1 for i, node in enumerate(order)}
    first = {value: position[node] for value, node in producer.items()}
    last = first.copy()
    for node, values in inputs.items():
        for value in values:
            first.setdefault(value, 0)
            last[value] = max(last.get(value, 0), position[node])
    returned = {
        value for value in {*first, *sinks} if value.startswith('"return_')
    }
    for value in sinks | returned:
        first.setdefault(value, 0)
        last[value] = len(order) + 1

    changes = [0] * (len(order) + 3)
    for value, begin in first.items():
        changes[begin] += 1
        changes[last[value] + 1] -= 1
    live = peak = 0
    for change in changes:
        live += change
        peak = max(peak, live)
    counts = Counter(kinds.values())
    return {
        "primitives": len(kinds),
        "primitive_counts": dict(counts.most_common()),
        "input_values": sum(begin == 0 for begin in first.values()),
        "returned_values": len(returned),
        "terminal_values": len(sinks),
        "values_with_multiple_uses": sum(
            uses > 1
            for uses in Counter(
                v for row in inputs.values() for v in row
            ).values()
        ),
        "peak_logical_live_values": peak,
    }


def _gemm(a, b, lhs, rhs, out):
    if len(set(lhs)) != len(lhs) or len(set(rhs)) != len(rhs):
        return None
    dims = dict(zip(lhs, a.shape))
    if any(n in dims and dims[n] != size for n, size in zip(rhs, b.shape)):
        return None
    dims.update(zip(rhs, b.shape))
    if (set(lhs) ^ set(rhs)) - set(out):
        return None
    batch = tuple(n for n in out if n in lhs and n in rhs)
    left = tuple(n for n in out if n in lhs and n not in rhs)
    right = tuple(n for n in out if n in rhs and n not in lhs)
    contracted = tuple(n for n in lhs if n not in out)
    if (
        not left
        or not right
        or not contracted
        or any(not s for s in dims.values())
    ):
        return None

    def shape(names):
        return tuple(dims[n] for n in names)

    m, n, k = (math.prod(shape(group)) for group in (left, right, contracted))
    pa = tuple(lhs.index(axis) for axis in (*batch, *left, *contracted))
    pb = tuple(rhs.index(axis) for axis in (*batch, *contracted, *right))
    a = a.transpose(pa).reshape(*shape(batch), m, k)
    b = b.transpose(pb).reshape(*shape(batch), k, n)
    order = (*batch, *left, *right)
    return (
        (a @ b)
        .reshape(shape(order))
        .transpose(tuple(order.index(i) for i in out))
    )


def _vocabulary_projection(a, b, lhs, rhs, out):
    # Restrict the probe to the historical recipe's forward projection.
    # Products for dW and dX have different input ranks and free axes.
    if b.ndim == 2:
        a, b, lhs, rhs = b, a, rhs, lhs
    if a.ndim != 2 or b.ndim != 3 or len(out) != 3:
        return False
    width = set(lhs) & set(rhs)
    if len(width) != 1 or set(out) != set(lhs) ^ set(rhs):
        return False
    axis = next(iter(width))
    return (
        a.shape[lhs.index(axis)] == b.shape[rhs.index(axis)] == 768
        and a.shape[1 - lhs.index(axis)] == 50304
        and sorted(size for label, size in zip(rhs, b.shape) if label != axis)
        in ([1, 1024], [2, 1024])
    )


def _install_contractions(policy, census):
    original = _Runtime.contract

    def contract(self, arrays, labels, output):
        if self.tracing and census is not None:
            signature = (
                tuple(
                    (tuple(a.shape), str(a.dtype), tuple(names))
                    for a, names in zip(arrays, labels)
                ),
                tuple(output),
            )
            census[signature] += 1
        if (
            self.tracing
            and len(arrays) == 2
            and all(self.owns(a) and a.dtype == mx.float32 for a in arrays)
        ):
            a, b = arrays
            lhs, rhs = labels
            if policy == "layout":
                result = _gemm(a, b, lhs, rhs, output)
                if result is not None:
                    self.counts["probe_layout_gemm"] += 1
                    return result
            if policy == "bf16" and _vocabulary_projection(
                a, b, lhs, rhs, output
            ):
                alphabet = string.ascii_letters

                def names(axes):
                    return "".join(alphabet[i] for i in axes)

                equation = f"{names(lhs)},{names(rhs)}->{names(output)}"
                self.counts["probe_vocabulary_bf16"] += 1
                return mx.einsum(
                    equation, a.astype(mx.bfloat16), b.astype(mx.bfloat16)
                ).astype(mx.float32)
        return original(self, arrays, labels, output)

    _Runtime.contract = contract


def _install_token_sum():
    original = _dense_indexed._add_indexed

    def add(runtime, result, items):
        if len(items) < 2:
            return original(runtime, result, items)
        part, (indexed, output, _) = items[0]
        key = indexed.key.get()
        if not runtime.owns(key):
            return None
        indexes, _ = runtime.indexes(output[1].values, key)
        keys, slots, active = _dense_indexed._coalesce(indexes)
        rows = mx.arange(len(output[0]))[:, None]
        tokens = indexes.reshape(-1)
        permutation = tuple(part.axes.index(a) for a in output)
        combined = mx.zeros((len(output[0]), tokens.size), dtype=mx.float32)
        for member, (indexed, output, mask) in items:
            data = indexed.value.get()
            if not runtime.owns(data) or data.dtype != mx.float32:
                return None
            data = runtime.broadcast(
                data,
                indexed.value.axes,
                indexed.local,
                tuple(len(a) for a in indexed.local),
            ).reshape(combined.shape)
            data = data * _dense_indexed._mask_at(runtime, mask, output, tokens)
            valid = _dense_indexed._mask_at(
                runtime, member.mask, output, tokens
            )
            combined = combined + mx.where(valid, data, 0)
        partial = mx.zeros_like(combined).at[rows, slots[None, :]].add(combined)
        result = (
            result.transpose(permutation)
            .at[rows, keys[None, :]]
            .add(mx.where(active, partial, 0))
        )
        runtime.counts["probe_token_sum_partials"] += len(items)
        runtime.counts["gpu_indexed_union_partials"] += len(items)
        return result.transpose(permutation)

    _dense_indexed._add_indexed = add


def _install_profile(path, update, policy, census):
    original = _Runtime.finish_training
    count = 0

    def finish(self, arrays, status):
        nonlocal count
        count += 1
        if count != update:
            return original(self, arrays, status)
        dot = path / "step.dot"
        outputs = [*arrays, *([status] if status is not None else [])]
        named = {f"return_{i}": array for i, array in enumerate(outputs)}
        mx.export_to_dot(str(dot), *outputs, **named)
        mx.reset_peak_memory()
        before = self.memory()
        started = time.perf_counter()
        original(self, arrays, status)
        elapsed = time.perf_counter() - started
        after = self.memory()
        profile = {
            "policy": policy,
            "compiled_execution": update,
            "finish_seconds": elapsed,
            "memory_before": before,
            "memory_after": after,
            "graph": _graph_summary(dot),
            "return_layouts": {
                name: {"shape": array.shape, "dtype": str(array.dtype)}
                for name, array in named.items()
            },
            "contraction_layouts": [
                {"operands": operands, "output_axes": out, "count": times}
                for (operands, out), times in census.most_common()
            ],
        }
        (path / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")

    _Runtime.finish_training = finish


def _capture_gradients(path):
    original = _SQLTrainer.__init__
    microstep = _SQLTrainer.microstep

    def initialize(self, *args, **kwargs):
        original(self, *args, **(kwargs | {"verify": True}))

    def step(self, *args, **kwargs):
        result = microstep(self, *args, **kwargs)
        if not path.exists():
            # Diagnostic-only device readback. Preserve the actual output axis
            # order and avoid materializing rectangular Arrow coordinate grids.
            values = {
                name: np.asarray(table.table.resolve(None, "g").tensor.get())
                for name, table in result._outputs.items()
                if name.endswith("_grad")
            }
            np.savez(path, loss=np.asarray(result.loss), **values)
        return result

    _SQLTrainer.__init__ = initialize
    _SQLTrainer.microstep = step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("reference", "layout", "bf16", "token-sum"),
        default="reference",
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-update", type=int, default=5)
    parser.add_argument("--raw-gradients", type=Path)
    probe, forwarded = parser.parse_known_args()
    if probe.profile_update < 1:
        parser.error("--profile-update must be positive")
    census = Counter() if probe.profile_dir else None
    if probe.policy in {"layout", "bf16"} or census is not None:
        _install_contractions(probe.policy, census)
    if probe.policy == "token-sum":
        if any(
            os.environ.get(flag) != "1"
            for flag in (
                "XQL_MLX_INDEXED_UNION",
                "XQL_MLX_SHARED_INDEXED_UNION",
            )
        ):
            parser.error("token-sum requires both indexed UNION flags")
        _install_token_sum()
    if probe.profile_dir:
        probe.profile_dir.mkdir(parents=True, exist_ok=False)
        _install_profile(
            probe.profile_dir, probe.profile_update, probe.policy, census
        )
    if probe.raw_gradients:
        if probe.raw_gradients.suffix != ".npz" or probe.raw_gradients.exists():
            parser.error("use a new .npz path for raw gradients")
        _capture_gradients(probe.raw_gradients)
    sys.argv = [str(_DRIVER), *forwarded]
    runpy.run_path(str(_DRIVER), run_name="__main__")
    if probe.profile_dir and not (probe.profile_dir / "profile.json").exists():
        parser.error("requested compiled execution was not reached")
    if probe.raw_gradients and not probe.raw_gradients.exists():
        parser.error("no SQL gradients were captured")


if __name__ == "__main__":
    main()
