"""Compare the unapplied MLX pointwise-inlining experiment in one process.

Apply ``pointwise-inline-experiment.patch`` first. This diagnostic replays
identical full-training-step inputs in interleaved groups; the model does
not progress during those timing groups. Ordinary 100-update training is
measured separately by ``03_shakespeare.py``.
"""

from __future__ import annotations

import argparse
import json
import runpy
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

from xarray_sql._dense_mlx import _Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("pointwise-replay.json")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/pointwise-replay-state")
    )
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.groups < 1 or args.repetitions < 1:
        parser.error("groups and repetitions must be positive")
    original = _Runtime.compile_training
    report = {}

    def compile_pair(runtime, function):
        if not hasattr(runtime, "inline_pointwise"):
            raise RuntimeError(
                "apply pointwise-inline-experiment.patch before this diagnostic"
            )
        runtime.counts["gpu_training_compilations"] += 1

        def make(inline):
            def traced(*inputs):
                runtime.inline_pointwise = inline
                return function(*inputs)

            return mx.compile(traced)

        variants = {"nested": make(False), "inline": make(True)}
        done = False

        def execute(*inputs):
            nonlocal done
            if not done:
                outputs = {}
                for name, variant in variants.items():
                    outputs[name] = variant(*inputs)
                    mx.eval(outputs[name])
                    for _ in range(3):
                        mx.eval(variant(*inputs))
                differences = []
                for a, b in zip(outputs["nested"][0], outputs["inline"][0]):
                    assert a.shape == b.shape and a.dtype == b.dtype
                    assert bool(mx.allclose(a, b, atol=3e-5, rtol=3e-5).item())
                    differences.append(float(mx.max(mx.abs(a - b)).item()))
                rows = []
                for group, name in enumerate(
                    ["nested", "inline", "inline", "nested"] * args.groups
                ):
                    variant = variants[name]
                    started = time.perf_counter()
                    for _ in range(args.repetitions):
                        packed, status = variant(*inputs)
                        mx.eval(packed, status)
                        assert bool(status.item())
                        float(packed[0].item())
                    rows.append(
                        {
                            "group": group,
                            "variant": name,
                            "seconds_per_step": (time.perf_counter() - started)
                            / args.repetitions,
                        }
                    )
                report.update(
                    {
                        "scope": "Fixed-input replay of a full forward/backward/optimizer graph; not a progressing training trajectory. ABBA groups in one process.",
                        "repetitions_per_group": args.repetitions,
                        "rows": rows,
                        "mean_seconds": {
                            name: statistics.mean(
                                r["seconds_per_step"]
                                for r in rows
                                if r["variant"] == name
                            )
                            for name in variants
                        },
                        "max_output_absolute_difference": max(differences),
                    }
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                done = True
            return variants["inline"](*inputs)

        return execute

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks/ml"))
    argv = sys.argv
    sys.argv = [
        str(root / "benchmarks/ml/03_shakespeare.py"),
        "--compile-training",
        "--gpu-dropout",
        "--discard-intermediates",
        "--freeze-context",
        "--steps",
        "1",
        "--eval-iters",
        "1",
        "--out-dir",
        str(args.out_dir),
    ]
    if args.data_dir is not None:
        sys.argv.extend(["--data-dir", str(args.data_dir)])
    _Runtime.compile_training = compile_pair
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        _Runtime.compile_training = original
        sys.argv = argv
    print(json.dumps(report))


if __name__ == "__main__":
    main()
