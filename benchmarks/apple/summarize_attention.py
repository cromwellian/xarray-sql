"""Summarize no-dropout attention experiments and optional local checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(path.read_text())


def _timing(rows):
    assert [r["iteration"] for r in rows] == list(range(100))
    warm = rows[1:]
    return {
        "updates": len(rows),
        "first_step_seconds": rows[0]["seconds"],
        "warm_step_seconds": statistics.mean(r["seconds"] for r in warm),
        "warm_step_median_seconds": statistics.median(
            r["seconds"] for r in warm
        ),
        "warm_input_and_step_seconds": statistics.mean(
            r["iteration_seconds"] for r in warm
        ),
        "total_timed_step_seconds": sum(r["seconds"] for r in rows),
    }


def _compare(reference, candidate):
    with np.load(reference) as ref, np.load(candidate) as got:
        keys = [k for k in ref.files if k.endswith(("_w", "_m", "_v"))]
        assert len(keys) == 33 and all(k in got for k in keys)
        errors = {}
        for key in keys:
            np.testing.assert_allclose(got[key], ref[key], rtol=3e-5, atol=3e-5)
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
    reference = _read(root / "attention-model-sql.json")["rows"]
    for row in reference:
        assert row["device_step"]["gpu_readback_bytes"] == 0
        assert row["device_step"]["training_exports"] == 0
        assert row["device_step"]["gpu_status_readbacks"] == 1
        assert row["device_step"]["gpu_loss_readbacks"] == 1
    runs = {"sql_ddx": _timing(reference)}
    comparisons = {}
    checkpoints = {}
    for variant in ["mlx-manual", "mlx-fast"]:
        data = _read(root / f"attention-model-{variant}.json")
        runs[variant] = _timing(data["rows"])
        checkpoints[variant] = root / f"attention-model-{variant}" / "last.npz"
        comparisons[variant] = max(
            abs(r["loss"] - c["loss"]) for r, c in zip(reference, data["rows"])
        )
    pair = _read(root / "attention-model-mlx-pair.json")
    for variant in ["mlx_manual", "mlx_fast"]:
        rows = [r for r in pair["rows"] if r["variant"] == variant]
        runs[f"paired_{variant}"] = _timing(rows)
        checkpoints[f"paired_{variant}"] = (
            root / "attention-model-mlx-pair" / f"{variant}-last.npz"
        )
        comparisons[f"paired_{variant}"] = max(
            abs(r["loss"] - c["loss"]) for r, c in zip(reference, rows)
        )
    assert max(comparisons.values()) < 3e-5
    kernel = _read(root / "attention-kernel-headroom.json")
    result = {
        "device": "Apple M5 Ultra, 30 CPU/64 GPU cores, 256 GiB, macOS 27.0.1",
        "mlx_version": kernel["mlx_version"],
        "dtype": kernel["dtype"],
        "dropout": 0,
        "parameters": pair["parameters"],
        "train_sha256": pair["train_sha256"],
        "xarray_sql_parent_commit": "0b3bc8419d47cfe31fdeea7ee2c6898a88a0e1bc",
        "einfold_commit": "ed0a982e661decdb6f36c6523f4e63ed472d7933",
        "runs": runs,
        "serial_manual_to_fast_speedup": runs["mlx-manual"]["warm_step_seconds"]
        / runs["mlx-fast"]["warm_step_seconds"],
        "paired_manual_to_fast_speedup": runs["paired_mlx_manual"][
            "warm_step_seconds"
        ]
        / runs["paired_mlx_fast"]["warm_step_seconds"],
        "maximum_loss_error_vs_sql": comparisons,
        "scope": "SQL uses ddx derivatives and full validation. Native MLX controls use autodiff and finite-value checks without SQL row/NULL validation. No-dropout full-model training, not an integrated SQL attention optimization. Warm steps 2-100; input/evaluation/logging/checkpoint I/O excluded.",
        "evidence_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                root / "attention-kernel-headroom.json",
                root / "attention-model-sql.json",
                root / "attention-model-mlx-manual.json",
                root / "attention-model-mlx-fast.json",
                root / "attention-model-mlx-pair.json",
            ]
        },
    }
    if args.check_checkpoints:
        result["checkpoints_vs_sql"] = {
            name: _compare(root / "attention-model-sql/last.npz", candidate)
            for name, candidate in checkpoints.items()
        }
    (root / "attention-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
