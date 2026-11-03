"""Diagnose float32 Adam sensitivity in the historical speedrun port.

At initialization the residual projections are zero, so the tied embedding
gradient has an independent, small float64 reference: embedding -> RMS
normalization -> tied output head -> cross entropy. This is a diagnostic,
not a timed training backend or a replacement for ddx gradients.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
    recipe = importlib.import_module("_modded_nanogpt")
    sql = importlib.import_module("_modded_sql")
    native = importlib.import_module("_modded_torch")
    config = recipe._Config()
    params = recipe._initial(config)
    pair = recipe._ShardStream(
        args.data_dir.glob("fineweb_train_*.bin"), config
    ).next()
    actual = sql._SQLTrainer(config, params, verify=True)
    expected = native._TorchTrainer(config, params, args.upstream, "mps")
    try:
        actual.schedule(0)
        expected.schedule(0)
        result = actual.microstep(*pair, final=True)
        expected_loss = expected.microstep(*pair, final=True)
        a, b = (
            actual.arrays(result, gradients=True),
            expected.arrays(gradients=True),
        )
        report = {
            "config": config.manifest(),
            "sql_loss": result.loss_value(),
            "torch_loss": expected_loss,
            "arrays": {},
            "selected": [],
        }
        for name in a:
            delta = np.abs(a[name] - b[name])
            report["arrays"][name] = {
                "max_absolute_error": float(delta.max()),
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "violations": int(
                    np.sum(delta > 3e-5 + 3e-5 * np.abs(b[name]))
                ),
            }
        for index in ((1807, 467), (416, 290), (3414, 60)):
            report["selected"].append(
                {
                    "index": index,
                    "initial": float(params["wte"][index]),
                    "sql": {
                        name: float(a[name][index])
                        for name in ("wte_w", "wte_m", "wte_v", "wte_g")
                    },
                    "torch": {
                        name: float(b[name][index])
                        for name in ("wte_w", "wte_m", "wte_v", "wte_g")
                    },
                }
            )
    finally:
        actual.close()
    del actual, expected, a, b
    weight = torch.tensor(
        params["wte"], dtype=torch.float64, requires_grad=True
    )
    tok, nxt = (torch.tensor(t, dtype=torch.long) for t in pair)
    hidden = F.embedding(tok, weight)
    hidden = hidden / torch.sqrt(
        torch.mean(hidden * hidden, dim=-1, keepdim=True)
        + np.finfo(np.float32).eps
    )
    logits = hidden @ weight.T
    loss = F.cross_entropy(logits.flatten(0, 1), nxt.flatten())
    loss.backward()
    report["float64_embedding_only_loss"] = float(loss.detach())
    for record in report["selected"]:
        record["float64_gradient"] = float(weight.grad[tuple(record["index"])])
    report["raw_gradient_violations"] = sum(
        e["violations"] for n, e in report["arrays"].items() if n.endswith("_g")
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "raw_gradient_violations": report["raw_gradient_violations"],
                "selected": report["selected"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
