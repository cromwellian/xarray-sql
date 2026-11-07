"""Attribute the training graph's contractions and logical tensor sizes.

Uses MLX's native no-fusion diagnostic mode because its exported Compiled
primitives have no serialization state in MLX 0.32.4. The simplifier remains
enabled. Timings from this process include the changed compilation policy.
Logical byte counts describe tensor extents, including views and broadcasts.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from xarray_sql._dense_mlx import _Runtime

_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _ROOT / "benchmarks/ml/04_modded_nanogpt.py"


def _spec(value):
    name, shape, dtype = value
    return {
        "id": name,
        "shape": shape,
        "dtype": str(dtype),
        "bytes": math.prod(shape) * mx.array([], dtype=dtype).itemsize,
    }


def _summarize(primitives):
    totals = defaultdict(
        lambda: {
            "count": 0,
            "logical_input_bytes": 0,
            "logical_output_bytes": 0,
        }
    )
    matmuls = defaultdict(
        lambda: {"count": 0, "estimated_flops": 0, "logical_output_bytes": 0}
    )
    for node in primitives:
        slot = totals[node["name"]]
        slot["count"] += 1
        slot["logical_input_bytes"] += sum(v["bytes"] for v in node["inputs"])
        slot["logical_output_bytes"] += sum(v["bytes"] for v in node["outputs"])
        if node["name"] != "Matmul":
            continue
        left, right = node["inputs"]
        output = node["outputs"][0]
        shape = (
            tuple(left["shape"]),
            tuple(right["shape"]),
            tuple(output["shape"]),
        )
        slot = matmuls[shape]
        slot["count"] += 1
        slot["estimated_flops"] += (
            2 * math.prod(output["shape"]) * left["shape"][-1]
        )
        slot["logical_output_bytes"] += output["bytes"]
    return {
        "primitive_count": len(primitives),
        "primitive_totals": dict(totals),
        "matmuls": [
            {"shapes": shape, **values}
            for shape, values in sorted(
                matmuls.items(), key=lambda item: -item[1]["estimated_flops"]
            )
        ],
    }


def _install_capture(directory, selected):
    original = _Runtime.finish_training
    execution = 0

    def finish(self, arrays, status):
        nonlocal execution
        execution += 1
        if execution != selected:
            return original(self, arrays, status)
        roots = [*arrays, *([status] if status is not None else [])]
        primitives, constants, outputs = [], [], []

        def callback(item):
            kind = item["type"]
            if kind == "primitive":
                primitives.append(
                    {
                        "name": item["name"],
                        "inputs": [_spec(v) for v in item["inputs"]],
                        "outputs": [_spec(v) for v in item["outputs"]],
                    }
                )
            elif kind == "constants":
                constants.extend(
                    {
                        "id": name,
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "bytes": value.nbytes,
                    }
                    for name, value in item["constants"]
                )
            elif kind == "outputs":
                outputs.extend(_spec(v) for v in item["outputs"])

        mx.export_function(callback, lambda: tuple(roots))
        report = {
            "compile_mode": "no_fuse",
            "mlx_version": importlib.metadata.version("mlx"),
            "compiled_execution": execution,
            **_summarize(primitives),
        }
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        (directory / "primitives.json").write_text(
            json.dumps(
                {
                    "primitives": primitives,
                    "constants": constants,
                    "outputs": outputs,
                },
                indent=2,
            )
            + "\n"
        )
        return original(self, arrays, status)

    _Runtime.finish_training = finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes-dir", required=True, type=Path)
    parser.add_argument("--shapes-update", type=int, default=5)
    args, forwarded = parser.parse_known_args()
    if args.shapes_update < 1:
        parser.error("--shapes-update must be positive")
    if importlib.metadata.version("mlx") != "0.32.4":
        parser.error("the native compilation mode binding requires MLX 0.32.4")
    args.shapes_dir.mkdir(parents=True, exist_ok=False)
    # MLX's public C++ API defines CompileMode::no_fuse as the third enum
    # entry. This binding is confined to the version-checked probe process.
    library = ctypes.CDLL(str(Path(mx.__file__).parent / "lib/libmlx.dylib"))
    set_mode = library._ZN3mlx4core16set_compile_modeENS0_11CompileModeE
    set_mode.argtypes = [ctypes.c_int]
    set_mode.restype = None
    set_mode(2)
    _install_capture(args.shapes_dir, args.shapes_update)
    sys.path.insert(0, str(_DRIVER.parent))
    sys.argv = [str(_DRIVER), *forwarded]
    runpy.run_path(str(_DRIVER), run_name="__main__")
    if not (args.shapes_dir / "summary.json").exists():
        parser.error("the requested compiled execution was not reached")


if __name__ == "__main__":
    main()
