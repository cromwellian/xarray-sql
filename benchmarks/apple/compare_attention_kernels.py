"""Compare resident SQL/ddx attention with MLX attention without dropout.

[ddx](https://github.com/xqlsystems/ddx) differentiates SQL and
[EinFold](https://github.com/xqlsystems/einfold) lowers it to compact arrays.
[MLX](https://github.com/ml-explore/mlx) autodiff is an independent control
only. Timings replay fixed inputs, not a progressing full-model training run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import xarray as xr

from xarray_sql import Context
from xarray_sql.training import TrainingSession


def _sql(head_size):
    return f"""WITH scores AS (
      SELECT q.b,q.h,q.i,k.j,SUM(q.val*k.val)/{math.sqrt(head_size)} AS s
      FROM q JOIN k ON q.b=k.b AND q.h=k.h AND q.c=k.c
      WHERE q.i>=k.j GROUP BY q.b,q.h,q.i,k.j),
    maximum AS (SELECT b,h,i,ddx_stop_gradient(MAX(s)) AS m FROM scores GROUP BY b,h,i),
    weights AS (SELECT s.b,s.h,s.i,s.j,exp(s.s-m.m) AS p FROM scores s
      JOIN maximum m ON s.b=m.b AND s.h=m.h AND s.i=m.i),
    denominator AS (SELECT b,h,i,SUM(p) AS z FROM weights GROUP BY b,h,i),
    attention AS (SELECT a.b,a.h,a.i,v.c,SUM(a.p/n.z*v.val) AS val
      FROM weights a JOIN denominator n ON a.b=n.b AND a.h=n.h AND a.i=n.i
      JOIN v ON a.b=v.b AND a.h=v.h AND a.j=v.j GROUP BY a.b,a.h,a.i,v.c),
    loss AS (SELECT SUM(a.val*d.val) AS l FROM attention a JOIN dy d
      ON a.b=d.b AND a.h=d.h AND a.i=d.i AND a.c=d.c)
    """


def _sql_program(values, backward):
    ctx = Context()
    dims = {
        "q": ("b", "h", "i", "c"),
        "k": ("b", "h", "j", "c"),
        "v": ("b", "h", "j", "c"),
        "dy": ("b", "h", "i", "c"),
    }
    for name, array in zip(["q", "k", "v", "dy"], values):
        ctx.from_dataset(
            name, xr.Dataset({"val": (dims[name], array.astype(np.float64))})
        )
    session = TrainingSession(ctx, max_cached_bytes=16 << 30)
    for name in dims:
        session.bind(name)
    root = _sql(values[0].shape[-1])
    names = ["q", "k", "v"] if backward else ["q"]
    gradients = {
        "g" + n: root
        + f"SELECT {','.join(dims[n])},grad(loss,val) AS val FROM {n} ORDER BY {','.join(dims[n])}"
        for n in names
    }
    updates = (
        {"next_" + n: f"SELECT * FROM g{n}" for n in names} if backward else {}
    )
    step = session.prepare_step(
        gradients=gradients,
        stages={},
        updates=updates,
        assignments={n: ("next_" + n, [*dims[n], "val"]) for n in names}
        if backward
        else {},
        compile=True,
        capture_intermediates=False,
        validation="none",
        freeze_context=True,
    )
    runtime = session.executor._runtime
    original = runtime.compile_training
    captured = {}

    def capture(function):
        compiled = original(function)
        captured["function"] = compiled

        def execute(*arrays):
            captured["inputs"] = arrays
            return compiled(*arrays)

        return execute

    runtime.compile_training = capture
    try:
        step()
    finally:
        runtime.compile_training = original
    slots = {
        name: captured["inputs"][i]
        for i, (name, _) in enumerate(step._input_slots)
    }

    def execute():
        packed, status = captured["function"](*captured["inputs"])
        assert status is None
        return packed

    return execute, [slots[n] for n in ["q", "k", "v", "dy"]], session


def _manual(q, k, v):
    length = q.shape[-2]
    scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(q.shape[-1])
    scores = mx.where(
        mx.arange(length)[:, None] >= mx.arange(length)[None, :],
        scores,
        -float("inf"),
    )
    weights = mx.exp(
        scores - mx.stop_gradient(mx.max(scores, axis=-1, keepdims=True))
    )
    return (weights / mx.sum(weights, axis=-1, keepdims=True)) @ v


def _fast(q, k, v):
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5, mask="causal", force_fused=True
    )


def _control(function, values, backward):
    def loss(q, k, v, dy):
        return mx.sum(function(q, k, v) * dy)

    differentiated = mx.value_and_grad(loss, argnums=(0, 1, 2))

    def execute(*args):
        if backward:
            value, gradients = differentiated(*args)
            return [value, *gradients]
        q, k, v, dy = args
        out = function(q, k, v)
        return [mx.sum(out * dy)]

    compiled = mx.compile(execute)
    return lambda: compiled(*values)


def _measure(functions, groups, repetitions, warmup):
    results = {}
    for name, function in functions.items():
        for _ in range(warmup):
            result = function()
            mx.eval(result)
        results[name] = result
    errors = {}
    details = {}
    reference = results["sql_ddx"]
    for name, result in results.items():
        assert len(result) == len(reference)
        maximum = 0.0
        absolute = []
        for actual, expected in zip(result, reference):
            a, e = np.asarray(actual), np.asarray(expected)
            assert a.shape == e.shape
            np.testing.assert_allclose(a, e, atol=3e-5, rtol=3e-5)
            maximum = max(maximum, float(np.max(np.abs(a - e))))
            absolute.append(float(np.max(np.abs(a - e))))
        errors[name] = maximum
        details[name] = dict(zip(["loss", "dq", "dk", "dv"], absolute))
    rows = []
    names = list(functions)
    for group in range(groups):
        order = names if group % 2 == 0 else names[::-1]
        for name in order:
            started = time.perf_counter()
            for _ in range(repetitions):
                mx.eval(functions[name]())
            rows.append(
                {
                    "group": group,
                    "variant": name,
                    "seconds": (time.perf_counter() - started) / repetitions,
                }
            )
    means = {
        n: statistics.mean(r["seconds"] for r in rows if r["variant"] == n)
        for n in names
    }
    return {
        "rows": rows,
        "mean_seconds": means,
        "max_absolute_error_vs_sql": errors,
        "absolute_error_by_output": details,
        "fast_vs_sql_speedup": means["sql_ddx"] / means["mlx_fast"],
        "fast_vs_manual_speedup": means["mlx_manual"] / means["mlx_fast"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        mx.default_device() != mx.gpu
        or os.environ.get("MLX_ENABLE_TF32") != "0"
    ):
        parser.error("requires MLX GPU and MLX_ENABLE_TF32=0")
    if os.environ.get("MLX_SDPA_VJP_FALLBACK", "0") != "0":
        parser.error(
            "clear MLX_SDPA_VJP_FALLBACK for the builtin backward measurement"
        )
    if args.groups < 1 or args.repetitions < 1 or args.warmup < 1:
        parser.error("groups, repetitions and warmup must be positive")
    shapes = [
        tuple(map(int, s.split(",")))
        for s in (args.shape or ["12,6,256,64", "12,12,256,64", "1,12,1024,64"])
    ]
    results = []
    for shape in shapes:
        if len(shape) != 4 or min(shape) < 1 or shape[2] < 32:
            parser.error(
                "shape must be positive B,H,T,D with T>=32 for this fused-training experiment"
            )
        rng = np.random.default_rng(1337)
        values = [rng.normal(size=shape).astype(np.float32) for _ in range(4)]
        entry = {"shape": shape}
        for backward in [False, True]:
            sql, inputs, owner = _sql_program(values, backward)
            output_errors = {}
            if not backward:
                output = owner.query(
                    _sql(shape[-1])
                    + "SELECT b,h,i,c,val FROM attention ORDER BY b,h,i,c"
                )
                expected = np.asarray(output["val"]).reshape(shape)
                for forward in [_manual, _fast]:
                    actual = np.asarray(forward(*inputs[:3]))
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=3e-5,
                        rtol=3e-5,
                    )
                    output_errors[forward.__name__] = float(
                        np.max(np.abs(actual - expected))
                    )
            functions = {
                "sql_ddx": sql,
                "mlx_manual": _control(_manual, inputs, backward),
                "mlx_fast": _control(_fast, inputs, backward),
            }
            before = owner.statistics().get("gpu_readback_bytes", 0)
            result = _measure(
                functions, args.groups, args.repetitions, args.warmup
            )
            assert owner.statistics().get("gpu_readback_bytes", 0) == before
            if output_errors:
                result["forward_output_max_absolute_error"] = output_errors
            result["sql_measurements"] = owner.statistics()
            entry["forward_and_qkv_gradients" if backward else "forward"] = (
                result
            )
            print(
                json.dumps({"shape": shape, "backward": backward, **result}),
                flush=True,
            )
            owner.close()
        results.append(entry)
    report = {
        "mlx_version": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "dtype": "float32, TF32 disabled",
        "dropout": 0,
        "warmup_calls_per_variant": args.warmup,
        "groups": args.groups,
        "repetitions_per_group": args.repetitions,
        "scope": "Fixed-input replay of attention plus scalar dot loss, and separately loss plus all Q/K/V gradients. No optimizer or full model; SQL gradients use ddx; MLX autodiff is a control only. SQL policy validation omitted for this numerical kernel comparison.",
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
