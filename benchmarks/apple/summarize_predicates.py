"""Rebuild the Boolean/presence and GPU layout training ablation evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from summarize_compiler import _checkpoint, _read

_RUNS = {
    "reference": ("predicates-reference", "off", False, False),
    "boolean": ("predicates-boolean", "boolean", False, False),
    "presence": ("predicates-presence", "presence", False, False),
    "all": ("predicates-all", "all", False, False),
    "direct": ("union-only", "off", True, False),
    "direct_boolean": ("union-boolean", "boolean", True, False),
    "direct_all": ("predicates-union-all", "all", True, False),
    "inline_all_experiment": ("union-inline-all", "all", True, True),
    "inline_boolean_experiment": (
        "union-inline-boolean",
        "boolean",
        True,
        True,
    ),
    "nested_all_repeat": ("union-nested-repeat", "all", True, False),
    "reference_repeat": ("predicates-repeat-reference", "off", False, False),
    "default_repeat": ("predicates-repeat-default", "boolean", True, False),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runs, samples = {}, []
    for label, (name, predicates, direct, inline) in _RUNS.items():
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
            step = row["device_step"]
            assert (
                step["gpu_readback_bytes"] == 0
                and step["training_exports"] == 0
            )
            assert (
                step["gpu_status_readbacks"] == 1
                and step["gpu_loss_readbacks"] == 1
            )
            samples.append(
                {
                    "run": label,
                    **{
                        k: row[k]
                        for k in [
                            "iteration",
                            "loss",
                            "seconds",
                            "input_seconds",
                            "iteration_seconds",
                        ]
                    },
                }
            )
        runs[label] = {
            "directory": name,
            "flags": {
                "predicates": predicates,
                "direct_union": direct,
                "experimental_inline": inline,
            },
            "config": log[0]["config"],
            "parameters": log[0]["parameters"],
            "first_step_seconds": rows[0]["seconds"],
            "warm_step_seconds": statistics.mean(
                r["seconds"] for r in rows[1:]
            ),
            "warm_step_median_seconds": statistics.median(
                r["seconds"] for r in rows[1:]
            ),
            "warm_iteration_seconds": statistics.mean(
                r["iteration_seconds"] for r in rows[1:]
            ),
            "total_step_seconds": sum(r["seconds"] for r in rows),
            "mlx_memory": rows[-1]["mlx_memory"],
            "measurements": log[-1]["measurements"],
        }
        if args.check_checkpoints and label != "reference":
            runs[label]["checkpoint_vs_reference"] = _checkpoint(
                root / _RUNS["reference"][0] / "last.npz",
                root / name / "last.npz",
            )
    comparisons = {}
    for a, b in [
        ("reference", "direct_boolean"),
        ("reference_repeat", "default_repeat"),
    ]:
        ref, new = runs[a], runs[b]
        assert {k: v for k, v in ref["config"].items() if k != "out_dir"} == {
            k: v for k, v in new["config"].items() if k != "out_dir"
        }
        comparisons[f"{a}_to_{b}"] = {
            "warm_speedup": ref["warm_step_seconds"] / new["warm_step_seconds"],
            "iteration_speedup": ref["warm_iteration_seconds"]
            / new["warm_iteration_seconds"],
            "total_100_step_speedup": ref["total_step_seconds"]
            / new["total_step_seconds"],
        }
    report = {
        "final_einfold_revision": "ed0a982e661decdb6f36c6523f4e63ed472d7933",
        "final_dense_program_sha256": "0d0257fd8e9a376bd06f8c0c83bcfca1a58f1e47f61a89db9a8cc02073efa1bf",
        "device": "Apple M5 Ultra, MLX 0.32.4 Metal, float32, TF32 disabled",
        "scope": "100 progressing training updates per run. Warm means exclude update 1, input preparation, evaluation and I/O. Full validation and GPU dropout enabled.",
        "runs": runs,
        "comparisons": comparisons,
    }
    (root / "predicate-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (root / "predicate-steps.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(samples[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
