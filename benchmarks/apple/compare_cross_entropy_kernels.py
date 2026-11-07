"""Isolate fused cross-entropy loss/backward on the trained model's logits.

Uses the historical Modded-NanoGPT control to produce real FP32 logits.
Only loss and its complete logits gradient are timed, with fixed inputs,
alternating kernel order, and explicit GPU synchronization. No optimizer
or complete model training speedup is measured by this probe.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from _modded_nanogpt import _Config, _ShardStream
from compare_modded_backward import _logits, _weights


def _function(fused):
    def loss(logits, targets):
        if fused:
            return mx.mean(mx.fast.cross_entropy(logits, targets))
        target = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return mx.mean(mx.logsumexp(logits, axis=-1) - target)

    return mx.compile(mx.value_and_grad(loss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        parser.error("requires MLX_ENABLE_TF32=0")
    if args.output.exists():
        parser.error("output path must be new")
    config = _Config()
    params = {
        name: mx.array(value)
        for name, value in _weights(args.weights, config).items()
    }
    tok, nxt = _ShardStream([args.data], config).next()
    logits = _logits(params, mx.array(tok), config, True)
    targets = mx.array(nxt)
    mx.eval(logits, targets)
    functions = {fused: _function(fused) for fused in [False, True]}
    samples, outputs = {False: [], True: []}, {}
    for fused in [False, True]:
        for _ in range(3):
            outputs[fused] = functions[fused](logits, targets)
            mx.eval(outputs[fused])
    for _ in range(6):
        for fused in [False, True, True, False]:
            start = time.perf_counter()
            outputs[fused] = functions[fused](logits, targets)
            mx.eval(outputs[fused])
            samples[fused].append(time.perf_counter() - start)
    left, right = (outputs[fused][1] for fused in [False, True])
    error = mx.abs(left - right)
    args.output.write_text(
        json.dumps(
            {
                "shape": logits.shape,
                "workload": "fixed real logits; loss and all logits gradients; no optimizer",
                "samples": {str(k): v for k, v in samples.items()},
                "median_seconds": {
                    str(k): statistics.median(v) for k, v in samples.items()
                },
                "loss": {
                    str(k): float(v[0].item()) for k, v in outputs.items()
                },
                "gradient_max_abs": float(mx.max(error).item()),
                "gradient_outside_tolerance": int(
                    mx.sum(error > 3e-5 + 3e-5 * mx.abs(left)).item()
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
