"""Summarize matched SQL compiler ablations and optional local checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

_RUNS = {
    "reference": "compiler-baseline-100",
    "facts": "compiler-facts-100",
    "cse": "compiler-cse-100",
    "combined": "compiler-all-100",
    "canonical_experiment": "compiler-canonical-100",
    "reference_repeat": "compiler-off-repeat",
    "cse_repeat": "compiler-cse-repeat",
    "final_cse": "compiler-final-cse",
    "gpu_reference": "compiler-gpu-off",
    "gpu_cse": "compiler-gpu-cse",
}


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _summary(root, name):
    log = _read(root / f"{name}.log")
    rows = [
        r
        for r in _read(root / name / "metrics.jsonl")
        if r.get("kind") == "train"
    ]
    assert len(rows) == 100 and [r["iteration"] for r in rows] == list(
        range(100)
    )
    for row in rows:
        assert row["device_step"]["gpu_readback_bytes"] == 0
        assert row["device_step"]["training_exports"] == 0
    return {
        "directory": name,
        "config": log[0]["config"],
        "parameters": log[0]["parameters"],
        "updates": len(rows),
        "warm_updates": len(rows) - 1,
        "first_step_seconds": rows[0]["seconds"],
        "warm_step_seconds": statistics.mean(r["seconds"] for r in rows[1:]),
        "warm_step_median_seconds": statistics.median(
            r["seconds"] for r in rows[1:]
        ),
        "warm_input_seconds": statistics.mean(
            r["input_seconds"] for r in rows[1:]
        ),
        "warm_iteration_seconds": statistics.mean(
            r["iteration_seconds"] for r in rows[1:]
        ),
        "mlx_memory": rows[-1]["mlx_memory"],
        "rss_bytes": rows[-1]["rss_bytes"],
        "measurements": log[-1]["measurements"],
        "optimizer_mode": "cse"
        if "cse" in name
        else (
            "all"
            if name in {"compiler-all-100", "compiler-canonical-100"}
            else "facts"
            if name == "compiler-facts-100"
            else "off"
        ),
        "metrics_sha256": hashlib.sha256(
            (root / name / "metrics.jsonl").read_bytes()
        ).hexdigest(),
    }


def _checkpoint(reference, candidate):
    with np.load(reference) as ref, np.load(candidate) as got:
        keys = [k for k in ref.files if k.endswith(("_w", "_m", "_v"))]
        assert keys
        maximum = 0.0
        for key in keys:
            np.testing.assert_allclose(got[key], ref[key], rtol=3e-5, atol=3e-5)
            maximum = max(maximum, float(np.max(np.abs(got[key] - ref[key]))))
    return {
        "arrays_checked": len(keys),
        "max_absolute_difference": maximum,
        "rtol": 3e-5,
        "atol": 3e-5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runs = {key: _summary(root, name) for key, name in _RUNS.items()}
    comparisons = {}
    for ref, new in [
        ("reference", "cse"),
        ("reference_repeat", "cse_repeat"),
        ("reference_repeat", "final_cse"),
        ("gpu_reference", "gpu_cse"),
    ]:
        a, b = runs[ref], runs[new]
        assert {k: v for k, v in a["config"].items() if k != "out_dir"} == {
            k: v for k, v in b["config"].items() if k != "out_dir"
        }
        entry = {
            "step_speedup": a["warm_step_seconds"] / b["warm_step_seconds"],
            "iteration_speedup": a["warm_iteration_seconds"]
            / b["warm_iteration_seconds"],
            "peak_allocation_reduction": 1
            - b["mlx_memory"]["peak_bytes"] / a["mlx_memory"]["peak_bytes"],
            "total_timed_step_seconds": {
                "reference": a["first_step_seconds"]
                + 99 * a["warm_step_seconds"],
                "candidate": b["first_step_seconds"]
                + 99 * b["warm_step_seconds"],
            },
        }
        if args.check_checkpoints:
            entry["checkpoint"] = _checkpoint(
                root / a["directory"] / "last.npz",
                root / b["directory"] / "last.npz",
            )
        comparisons[f"{ref}_to_{new}"] = entry
    result = {
        "final_source": {
            "einfold_commit": "23fc63d71badce02429e825ce3e1a37ce4bfdec7",
            "dense_program_sha256": "b53dc76ca88448977a82bbcc95d9851f67d5ca5cdeb8098817c9161e183d2fc8",
        },
        "precision": "float32, MLX_ENABLE_TF32=0",
        "device": "Apple M5 Ultra, MLX 0.32.4 Metal GPU",
        "timing": "Updates 2-100; synchronized loss, weights and moments. Input+step excludes evaluation, logging and checkpoint I/O.",
        "runs": runs,
        "comparisons": comparisons,
    }
    if args.check_checkpoints:
        result["additional_checkpoint_checks"] = {
            key: _checkpoint(
                root / _RUNS["reference"] / "last.npz",
                root / _RUNS[key] / "last.npz",
            )
            for key in ["facts", "combined", "canonical_experiment"]
        }
    (root / "compiler-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (root / "compiler-steps.csv").open("w", newline="") as output:
        fields = [
            "run",
            "iteration",
            "seconds",
            "input_seconds",
            "iteration_seconds",
            "loss",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, name in _RUNS.items():
            for row in _read(root / name / "metrics.jsonl"):
                if row.get("kind") == "train":
                    writer.writerow(
                        {"run": key, **{f: row[f] for f in fields[1:]}}
                    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
