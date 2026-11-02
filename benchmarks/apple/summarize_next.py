"""Summarize recorded follow-up GPU experiments and optionally check states."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    names = [
        "baseline",
        "pruned",
        "index",
        "index-repeat",
        "pruned-repeat",
        "matmul",
        "index-matmul",
        "buffer-tuning",
    ]
    output = {
        "date": "2026-11-02",
        "warm_scope": "Mean updates 2–100; 10.7M parameters, batch 12, context 256, float32, GPU dropout; MLX_ENABLE_TF32=0. Single serial run per row, not a confidence interval.",
        "runs": {},
    }
    for name in names:
        rows = [
            json.loads(l)
            for l in (root / ("next-" + name) / "metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        rows = [r for r in rows if r["kind"] == "train"]
        assert len(rows) == 100
        output["runs"][name] = {
            "steps": len(rows),
            "warm_seconds": statistics.mean(r["seconds"] for r in rows[1:]),
            "first_seconds": rows[0]["seconds"],
            "last_memory": rows[-1]["mlx_memory"],
            "timed_array_readback_bytes": sum(
                r["device_step"]["gpu_readback_bytes"] for r in rows
            ),
            "timed_exports": sum(
                r["device_step"]["training_exports"] for r in rows
            ),
        }
    output["paired_index_speedups"] = [
        output["runs"][a]["warm_seconds"] / output["runs"][b]["warm_seconds"]
        for a, b in [("pruned", "index"), ("pruned-repeat", "index-repeat")]
    ]
    if args.check_checkpoints:
        checks = {}
        for name in names[1:] + ["production-checked"]:
            reference = (
                "verified" if name == "production-checked" else "baseline"
            )
            a = np.load(
                root / ("next-" + reference) / "last.npz", allow_pickle=False
            )
            b = np.load(
                root / ("next-" + name) / "last.npz", allow_pickle=False
            )
            error = 0.0
            for key in a.files:
                if key == "metadata":
                    continue
                np.testing.assert_allclose(b[key], a[key], atol=3e-5, rtol=3e-5)
                error = max(error, float(np.max(np.abs(b[key] - a[key]))))
            checks[name] = {
                "reference": reference,
                "all_weights_and_moments_match": True,
                "maximum_absolute_error": error,
            }
        output["checkpoint_checks"] = checks
    (root / "next-summary.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
