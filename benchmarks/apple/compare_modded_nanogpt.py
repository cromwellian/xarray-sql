"""Compare matched historical Modded-NanoGPT SQL and PyTorch run directories.

The Modded-NanoGPT speedrun (https://github.com/KellerJordan/modded-nanogpt)
uses time to FineWeb validation loss. These local measurements compare the
same pinned float32 port on one device and never rank it against H100 records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _compare(left, right, checkpoints=False):
    a, b = [json.loads((p / "summary.json").read_text()) for p in (left, right)]
    keys = (
        "layers",
        "heads",
        "width",
        "vocab",
        "context",
        "batch",
        "accumulation",
        "iterations",
        "warmdown",
        "learning_rate",
        "ns_steps",
        "seed",
        "steps",
        "warmup_updates",
        "precision",
        "upstream_source_sha256",
        "initial_state_sha256",
        "implementation_sha256",
        "platform",
        "hardware",
        "data",
        "val_tokens",
        "target_loss",
    )
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f"runs differ in {key}")
    if a["oracle"] or b["oracle"]:
        raise ValueError(
            "use separate runs without oracle work for performance comparisons"
        )
    shards = lambda value: [
        (p["bytes"], p["sha256"]) for p in value["data_shards"]
    ]
    if shards(a) != shards(b):
        raise ValueError("runs used different ordered data shards")
    samples = []
    for directory, summary in ((left, a), (right, b)):
        rows = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        rows = [r for r in rows if r["kind"] == "train"]
        if [r["iteration"] for r in rows] != list(range(summary["steps"])):
            raise ValueError("incomplete or repeated update sequence")
        samples.append(rows)
    losses = [np.array([r["loss"] for r in rows]) for rows in samples]
    result = {
        "left": str(left),
        "right": str(right),
        "parameters": a["parameters"],
        "steps": a["steps"],
        "tokens_per_update": a["tokens_per_update"],
        "official_speedrun_comparable": False,
        "matched_configuration": True,
        "left_backend": a["backend"],
        "right_backend": b["backend"],
        "left_warm_ms": a["warm_seconds_per_update"] * 1000,
        "right_warm_ms": b["warm_seconds_per_update"] * 1000,
        "left_over_right_warm_time": a["warm_seconds_per_update"]
        / b["warm_seconds_per_update"],
        "left_training_seconds": a["training_seconds"],
        "right_training_seconds": b["training_seconds"],
        "maximum_loss_absolute_difference": float(
            np.max(np.abs(losses[0] - losses[1]))
        ),
        "losses_match_3e_5": bool(
            np.allclose(losses[0], losses[1], atol=3e-5, rtol=3e-5)
        ),
        "checkpoint_match_3e_5": None,
        "left_summary": a,
        "right_summary": b,
        "left_training_samples": samples[0],
        "right_training_samples": samples[1],
    }
    if checkpoints:
        errors = {}
        with (
            np.load(left / "last.npz", allow_pickle=False) as x,
            np.load(right / "last.npz", allow_pickle=False) as y,
        ):
            names = set(x.files) - {"metadata"}
            if names != set(y.files) - {"metadata"}:
                raise ValueError("checkpoint array names differ")
            for name in sorted(names):
                u, v = x[name], y[name]
                if (
                    u.shape != v.shape
                    or not np.isfinite(u).all()
                    or not np.isfinite(v).all()
                ):
                    raise ValueError(f"invalid checkpoint array: {name}")
                delta = np.abs(u - v)
                errors[name] = {
                    "max_absolute_error": float(delta.max()),
                    "rmse": float(np.sqrt(np.mean(delta * delta))),
                    "mismatching_elements": int(
                        np.sum(delta > 3e-5 + 3e-5 * np.abs(v))
                    ),
                    "elements": int(delta.size),
                }
        result["checkpoint_errors"] = errors
        result["checkpoint_match_3e_5"] = all(
            e["mismatching_elements"] == 0 for e in errors.values()
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--checkpoints", action="store_true")
    p.add_argument("--require-state-match", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.require_state_match and not args.checkpoints:
        p.error("--require-state-match requires --checkpoints")
    result = _compare(args.left, args.right, args.checkpoints)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if not k.endswith(("summary", "errors", "samples"))
            }
        ),
        flush=True,
    )
    if args.require_state_match and not result["checkpoint_match_3e_5"]:
        raise SystemExit(
            "strict checkpoint comparison failed; see saved per-array errors"
        )


if __name__ == "__main__":
    main()
