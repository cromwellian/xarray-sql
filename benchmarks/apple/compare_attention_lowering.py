"""Interleave generic and specialized SQL attention with identical inputs.

Both paths use ddx (https://github.com/xqlsystems/ddx) derivatives. This is
fixed-input kernel replay without an optimizer, not full-model training.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from compare_attention_kernels import _sql_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.groups < 1
        or args.repetitions < 1
        or os.environ.get("MLX_ENABLE_TF32") != "0"
        or mx.default_device() != mx.gpu
    ):
        parser.error(
            "requires positive counts and MLX GPU with MLX_ENABLE_TF32=0"
        )
    results = []
    for shape in [(12, 6, 256, 64), (12, 12, 256, 64), (1, 12, 1024, 64)]:
        rng = np.random.default_rng(1337)
        values = [rng.normal(size=shape).astype(np.float32) for _ in range(4)]
        for backward in [False, True]:
            functions, owners = {}, {}
            try:
                for name, flag in [("generic", "0"), ("lowered", "1")]:
                    os.environ["XQL_MLX_ATTENTION"] = flag
                    function, _, owner = _sql_program(values, backward)
                    functions[name], owners[name] = function, owner
                outputs = {}
                for name, fn in functions.items():
                    for _ in range(30):
                        outputs[name] = fn()
                        mx.eval(outputs[name])
                errors = []
                for a, e in zip(outputs["lowered"], outputs["generic"]):
                    a, e = np.asarray(a), np.asarray(e)
                    np.testing.assert_allclose(a, e, atol=3e-5, rtol=3e-5)
                    errors.append(float(np.max(np.abs(a - e))))
                before = {
                    n: o.statistics().get("gpu_readback_bytes", 0)
                    for n, o in owners.items()
                }
                samples = []
                names = list(functions)
                for group in range(args.groups):
                    for name in names if group % 2 == 0 else names[::-1]:
                        started = time.perf_counter()
                        for _ in range(args.repetitions):
                            mx.eval(functions[name]())
                        samples.append(
                            {
                                "group": group,
                                "variant": name,
                                "seconds": (time.perf_counter() - started)
                                / args.repetitions,
                            }
                        )
                for n, o in owners.items():
                    assert (
                        o.statistics().get("gpu_readback_bytes", 0) == before[n]
                    )
                means = {
                    n: statistics.mean(
                        x["seconds"] for x in samples if x["variant"] == n
                    )
                    for n in names
                }
                result = {
                    "shape": shape,
                    "phase": "forward_and_qkv_gradients"
                    if backward
                    else "forward",
                    "samples": samples,
                    "mean_seconds": means,
                    "speedup": means["generic"] / means["lowered"],
                    "absolute_error_by_output": dict(
                        zip(["loss", "dq", "dk", "dv"], errors)
                    ),
                    "statistics": {
                        n: o.statistics() for n, o in owners.items()
                    },
                }
                results.append(result)
                print(
                    json.dumps(
                        {
                            "shape": shape,
                            "backward": backward,
                            "means": means,
                            "speedup": result["speedup"],
                        }
                    ),
                    flush=True,
                )
            finally:
                for owner in owners.values():
                    owner.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "warmups": 30,
                "groups": args.groups,
                "repetitions": args.repetitions,
                "scope": "Same-process alternating-order fixed-input SQL/ddx attention replay. No optimizer or full model; validation policy omitted equally for kernel comparison. Both use the same group-proof setting.",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
