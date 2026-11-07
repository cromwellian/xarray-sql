"""Experimental code generation for unmasked FP32 row reductions.

Generates scalar expressions from the compact tensor graph, then uses a
fixed parallel reduction template. There are no model or operator-sequence
names in eligibility. FP32 addition order changes, so this is a diagnostic
probe with ordinary MLX lowering as fallback, not a default execution policy.
"""

from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from einfold.dense_program import _is

from xarray_sql._dense_mlx import _Runtime


def _generate(runtime, tensor):
    meta = tensor.reduction
    if meta is None or meta.kind != "sum" or not _is(meta.mask, True):
        return None
    axes = meta.value.axes
    reduced = tuple(a for a in meta.row_axes if a not in meta.kept)
    if not axes or reduced != axes[-1:] or set(axes) != set(meta.row_axes):
        return None
    shape = tuple(len(a) for a in axes)
    if shape[-1] < 32 or not math.prod(shape):
        return None
    leaves, seen = [], {}

    def expression(node):
        if node in seen:
            return seen[node]
        if len(seen) > 32:
            raise ValueError("expression too large")
        if node.kind in {"+", "-", "*", "/"}:
            text = f"({expression(node.args[0])} {node.kind} {expression(node.args[1])})"
        elif node.kind in {"sqrt", "exp", "neg"}:
            child = expression(node.args[0])
            text = (
                f"(-{child})"
                if node.kind == "neg"
                else f"metal::{node.kind}({child})"
            )
        elif node.kind == "array" and not node.axes:
            value = np.asarray(node.value)
            if value.dtype.kind not in "fiu" or not np.isfinite(value):
                raise ValueError("non-numerical constant")
            text = f"float({float(np.float32(value))!r})"
        else:
            if node.axes != axes:
                raise ValueError("broadcast or permuted leaf")
            value = node.get()
            if not runtime.owns(value) or value.dtype != mx.float32:
                raise ValueError("requires FP32 GPU leaves")
            text = f"in{len(leaves)}[row * {shape[-1]} + col]"
            leaves.append(value)
        seen[node] = text
        return text

    try:
        formula = expression(meta.value)
    except ValueError:
        return None
    if not leaves:
        return None
    source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        float value = 0.0f;
        for (uint col = tid; col < {shape[-1]}; col += 256) {{
            value += {formula};
        }}
        value = metal::simd_sum(value);
        threadgroup float partial[8];
        if ((tid & 31) == 0) partial[tid >> 5] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {{
            value = metal::simd_sum(tid < 8 ? partial[tid] : 0.0f);
            if (tid == 0) out[row] = value;
        }}
    """
    key = ("generated_row_sum", source, len(leaves))
    if key not in runtime.compiled:
        runtime.compiled[key] = mx.fast.metal_kernel(
            name="generated_row_sum",
            input_names=[f"in{i}" for i in range(len(leaves))],
            output_names=["out"],
            source=source,
            compile_options={"math_mode": "safe"},
        )
    result = runtime.compiled[key](
        inputs=leaves,
        grid=(math.prod(shape[:-1]) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape[:-1]],
        output_dtypes=[mx.float32],
    )[0]
    runtime.counts["gpu_generated_row_reductions"] += 1
    order = axes[:-1]
    return result.transpose(tuple(order.index(a) for a in tensor.axes))


def _install():
    original = _Runtime.reduction

    def reduction(self, tensor):
        value = _generate(self, tensor)
        return original(self, tensor) if value is None else value

    _Runtime.reduction = reduction


if __name__ == "__main__":
    _install()
    driver = Path(__file__).resolve().parents[1] / "ml/04_modded_nanogpt.py"
    sys.path.insert(0, str(driver.parent))
    runpy.run_path(str(driver), run_name="__main__")
