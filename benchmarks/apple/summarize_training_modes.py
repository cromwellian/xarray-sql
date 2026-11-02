"""Summarize validation/frozen-mode and matched JAX–MLX measurements."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def _rows(path):
    return [
        r
        for line in path.read_text().splitlines()
        if (r := json.loads(line))["kind"] == "train"
    ]


def _summary(rows):
    return {
        "updates": len(rows),
        "first_seconds": rows[0]["seconds"],
        "warm_seconds": statistics.mean(r["seconds"] for r in rows[1:]),
        "warm_input_seconds": statistics.mean(
            r["input_seconds"] for r in rows[1:]
        )
        if "input_seconds" in rows[0]
        else None,
        "warm_iteration_seconds": statistics.mean(
            r["iteration_seconds"] for r in rows[1:]
        )
        if "iteration_seconds" in rows[0]
        else None,
        "status_reads": sum(
            r.get("device_step", {}).get("gpu_status_readbacks", 0)
            for r in rows
        ),
        "array_readback_bytes": (
            sum(r["device_step"]["gpu_readback_bytes"] for r in rows)
            if "device_step" in rows[0]
            else None
        ),
    }


def _compare(left, right):
    a, b = np.load(left, allow_pickle=False), np.load(right, allow_pickle=False)
    worst = 0.0
    for name in b.files:
        if name == "metadata":
            continue
        np.testing.assert_allclose(a[name], b[name], rtol=3e-5, atol=3e-5)
        worst = max(worst, float(np.max(np.abs(a[name] - b[name]))))
    return {
        "all_weights_and_moments_match": True,
        "maximum_absolute_error": worst,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    modes = [
        "full",
        "state",
        "none",
        "full-frozen",
        "none-frozen",
        "final-full-frozen",
        "final-none-frozen",
    ]
    result = {
        "date": "2026-11-02",
        "model": "10,745,088 parameters, batch12/context256, exact GELU, dropout .2, clipped AdamW",
        "modes": {},
        "jax_control": {},
    }
    for mode in modes:
        result["modes"][mode] = _summary(
            _rows(root / ("modes-" + mode) / "metrics.jsonl")
        )
    for jax_name, sql_name in [
        ("jax-mlx-hundred", "jax-final-sql-hundred"),
        ("jax-mlx-repeat", "jax-sql-repeat"),
    ]:
        jax = json.loads((root / (jax_name + ".json")).read_text())
        sql = _summary(_rows(root / sql_name / "metrics.jsonl"))
        result["jax_control"][jax_name] = {
            "devices": jax["devices"],
            "packages": jax["packages"],
            "jax": _summary(jax["rows"]),
            "sql": sql,
            "sql_over_jax": sql["warm_seconds"] / jax["warm_seconds"],
        }
    if args.check_checkpoints:
        checks = {}
        for mode in modes[1:]:
            checks[mode] = _compare(
                root / "modes-full/last.npz",
                root / ("modes-" + mode) / "last.npz",
            )
        for name, sql in [
            ("five", "jax-matched-sql-five"),
            ("hundred", "jax-final-sql-hundred"),
            ("repeat", "jax-sql-repeat"),
        ]:
            checks["jax-" + name] = _compare(
                root / sql / "last.npz",
                Path("/private/tmp") / ("xql-jax-mlx-" + name + ".npz"),
            )
        result["checkpoint_checks"] = checks
    (root / "training-modes-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
