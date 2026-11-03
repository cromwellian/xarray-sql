"""Summarize actual SQL attention/group ablations and optional checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

_RUNS = {
    "reference": ("attention-final-reference", "0", "0"),
    "groups": ("attention-final-groups", "1", "0"),
    "forward": ("attention-final-forward", "1", "forward"),
    "all": ("attention-final-all", "1", "1"),
    "reference_repeat": ("attention-repeat-reference", "0", "0"),
    "all_repeat": ("attention-repeat-all", "1", "1"),
    "dropout_reference": ("groups-dropout-reference", "0", "0"),
    "dropout_selected": ("groups-dropout-selected", "1", "1"),
    "pinned_final": ("attention-pinned-final", "1", "1"),
    "auxiliary_experiment": ("attention-aux-model", "1", "1"),
}


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _checkpoint(reference, candidate):
    with np.load(reference) as ref, np.load(candidate) as got:
        keys = [k for k in ref.files if k.endswith(("_w", "_m", "_v"))]
        assert len(keys) == 33
        errors = {}
        for key in keys:
            np.testing.assert_allclose(got[key], ref[key], atol=3e-5, rtol=3e-5)
            errors[key] = float(np.max(np.abs(got[key] - ref[key])))
    return {
        "arrays_checked": len(keys),
        "atol": 3e-5,
        "rtol": 3e-5,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runs, rows = {}, []
    for key, (directory, groups, attention) in _RUNS.items():
        events = _read(root / f"{directory}.log")
        path = root / directory / "metrics.jsonl"
        steps = [r for r in _read(path) if r.get("kind") == "train"]
        assert [r["iteration"] for r in steps] == list(range(100))
        for row in steps:
            measured = row["device_step"]
            assert (
                measured["gpu_readback_bytes"]
                == measured["training_exports"]
                == 0
            )
            assert (
                measured["gpu_status_readbacks"]
                == measured["gpu_loss_readbacks"]
                == 1
            )
            rows.append({"run": key, **row})
        runs[key] = {
            "directory": directory,
            "config": events[0]["config"],
            "parameters": events[0]["parameters"],
            "flags": {
                "EINFOLD_DENSE_GROUP_PROOFS": groups,
                "XQL_MLX_ATTENTION": attention,
                "XQL_MLX_PRIMAL_AUX": "1"
                if key == "auxiliary_experiment"
                else "0",
            },
            "first_step_seconds": steps[0]["seconds"],
            "warm_step_seconds": statistics.mean(
                s["seconds"] for s in steps[1:]
            ),
            "warm_input_and_step_seconds": statistics.mean(
                s["iteration_seconds"] for s in steps[1:]
            ),
            "total_timed_step_seconds": sum(s["seconds"] for s in steps),
            "last_memory": steps[-1]["mlx_memory"],
            "measurements": events[-1]["measurements"],
            "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    comparisons = {}
    for reference, candidate in [
        ("reference", "groups"),
        ("groups", "forward"),
        ("groups", "all"),
        ("reference", "all"),
        ("reference_repeat", "all_repeat"),
        ("reference_repeat", "pinned_final"),
        ("dropout_reference", "dropout_selected"),
        ("pinned_final", "auxiliary_experiment"),
    ]:
        ref, got = runs[reference], runs[candidate]
        entry = {
            "step_speedup": ref["warm_step_seconds"] / got["warm_step_seconds"],
            "input_and_step_speedup": ref["warm_input_and_step_seconds"]
            / got["warm_input_and_step_seconds"],
            "total_timed_step_speedup": ref["total_timed_step_seconds"]
            / got["total_timed_step_seconds"],
        }
        if args.check_checkpoints:
            entry["checkpoint"] = _checkpoint(
                root / ref["directory"] / "last.npz",
                root / got["directory"] / "last.npz",
            )
        comparisons[f"{reference}_to_{candidate}"] = entry
    result = {
        "device": "Apple M5 Ultra, 30 CPU/64 GPU cores, 256 GiB, macOS 27.0.1",
        "mlx": "0.32.4, float32, MLX_ENABLE_TF32=0",
        "einfold_commit": "805f50ca6f57b9249815c3d5785e2c78724ef789",
        "xarray_sql_runtime_commit": "e52da15",
        "einfold_dense_program_sha256": "f80342023c23e60ced484b7f1239312273a4cb70a9c31582efd5c9a37e4979ea",
        "timing": "100 progressing SQL training updates. Warm means 2-100. Full validation, ddx derivatives, synchronization, loss/status scalar reads included; input, evaluation, logging and I/O excluded from step. Serial runs; no confidence intervals.",
        "runs": runs,
        "comparisons": comparisons,
    }
    (root / "attention-lowering-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    fields = [
        "run",
        "iteration",
        "loss",
        "seconds",
        "input_seconds",
        "iteration_seconds",
    ]
    with (root / "attention-lowering-steps.csv").open(
        "w", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
