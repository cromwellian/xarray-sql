"""No-dropout whole-model attention headroom controls.

SQL uses the existing Shakespeare generator with dropout operators omitted,
and ddx (https://github.com/xqlsystems/ddx) still supplies its derivatives.
Independent MLX (https://github.com/ml-explore/mlx) controls use autodiff and
differ only in generic versus builtin fused attention. They are not SQL
backend substitutions. All variants train the 10.7M character model from scratch.
The paired control alternates execution order while advancing both models
through identical batches, with separate weights and optimizer state.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def _initial(vocab):
    source = Path(__file__).parents[1] / "ml" / "02_nanogpt.py"
    tree = ast.parse(source.read_text())
    nodes = [
        n
        for n in tree.body
        if (
            isinstance(n, ast.FunctionDef)
            and n.name in {"init_params", "_parameter_sizes"}
        )
        or (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "PARAMS" for t in n.targets
            )
        )
    ]
    scope = {
        "np": np,
        "LAYERS": 6,
        "EMBED": 384,
        "HEADS": 6,
        "HEAD": 64,
        "HIDDEN": 1536,
        "CONTEXT": 256,
    }
    exec(  # noqa: S102 -- Trusted repository initializer only.
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
        scope,
    )
    return scope["init_params"](vocab, np.random.default_rng(1337))


def _norm(x, gain):
    centered = x - mx.mean(x, axis=-1, keepdims=True)
    return (
        centered
        / mx.sqrt(
            mx.mean(mx.power(centered, 2.0), axis=-1, keepdims=True) + 1e-5
        )
        * gain
    )


def _loss(p, tok, nxt, fast):
    x = p["wte"][tok] + p["wpe"][None]
    causal = mx.arange(256)[:, None] >= mx.arange(256)[None, :]
    for layer in range(6):
        h = _norm(x, p["ln_1"][layer])
        q, k, v = [
            mx.einsum("bte,ehc->bhtc", h, p[name][layer])
            for name in ["wq", "wk", "wv"]
        ]
        if fast:
            heads = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=0.125, mask="causal", force_fused=True
            )
        else:
            scores = (q @ k.swapaxes(-1, -2)) / 8
            scores = mx.where(causal, scores, -float("inf"))
            a = mx.exp(
                scores
                - mx.stop_gradient(mx.max(scores, axis=-1, keepdims=True))
            )
            heads = (a / mx.sum(a, axis=-1, keepdims=True)) @ v
        x = x + mx.einsum("bhtc,hce->bte", heads, p["wo"][layer])
        h = _norm(x, p["ln_2"][layer]) @ p["w_fc"][layer]
        hidden = 0.5 * h * (1 + mx.erf(h / math.sqrt(2)))
        x = x + hidden @ p["w_proj"][layer]
    logits = _norm(x, p["ln_f"]) @ p["wte"].T
    target = mx.take_along_axis(logits, nxt[..., None], axis=-1)[..., 0]
    return mx.mean(mx.logsumexp(logits, axis=-1) - target)


def _step_function(fast):
    differentiated = mx.value_and_grad(lambda p, t, n: _loss(p, t, n, fast))

    def step(p, m, v, tok, nxt, lr, b1, b2):
        loss, grads = differentiated(p, tok, nxt)
        norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()))
        clip = mx.minimum(1.0, 1.0 / (norm + 1e-6))
        new, nm, nv = {}, {}, {}
        status = mx.isfinite(loss) & mx.isfinite(clip)
        for name, w in p.items():
            status = status & mx.all(mx.isfinite(grads[name]))
            g = grads[name] * clip
            nm[name] = 0.9 * m[name] + 0.1 * g
            nv[name] = 0.99 * v[name] + 0.01 * g * g
            decay = 0.0 if name.startswith("ln_") else 0.1
            new[name] = w * (1 - lr * decay) - lr * (nm[name] / b1) / (
                mx.sqrt(nv[name] / b2) + 1e-8
            )
            for value in [new[name], nm[name], nv[name]]:
                status = status & mx.all(mx.isfinite(value))
        return new, nm, nv, loss, status

    return mx.compile(step)


def _sql(args):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks/ml"))
    source = root / "benchmarks/ml/03_shakespeare.py"
    spec = importlib.util.spec_from_file_location(
        "_no_dropout_sql_recipe", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._MODEL.model

    def no_dropout(**kwargs):
        return original(**{**kwargs, "dropout": False})

    module._MODEL.model = no_dropout
    module._masks = lambda arguments, rng: {}
    previous = sys.argv
    sys.argv = [
        str(source),
        "--compile-training",
        "--discard-intermediates",
        "--validation",
        "full",
        "--freeze-context",
        "--dropout",
        "0",
        "--steps",
        str(args.steps),
        "--log-interval",
        "1",
        "--profile-memory",
        "--out-dir",
        str(args.out_dir),
        "--data-dir",
        str(args.data_dir),
    ]
    try:
        module.main()
    finally:
        sys.argv = previous
        module._MODEL.model = original
    rows = [
        json.loads(line)
        for line in (args.out_dir / "metrics.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r.get("kind") == "train"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "backend": "sql_ddx",
                "dropout": 0,
                "mask_operators_omitted": True,
                "rows": rows,
                "warm_seconds": statistics.mean(r["seconds"] for r in rows[1:]),
                "checkpoint": str(args.out_dir / "last.npz"),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["sql", "mlx_manual", "mlx_fast", "mlx_pair"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "nanoGPT/data/shakespeare_char",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.steps <= 2001:
        parser.error("steps must be in [2,2001]")
    if (
        os.environ.get("MLX_ENABLE_TF32") != "0"
        or mx.default_device() != mx.gpu
    ):
        parser.error("requires MLX GPU and MLX_ENABLE_TF32=0")
    if args.backend == "sql":
        _sql(args)
        return
    if os.environ.get("MLX_SDPA_VJP_FALLBACK", "0") != "0":
        parser.error(
            "clear MLX_SDPA_VJP_FALLBACK for the builtin backward control"
        )
    train = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")
    vocab = len(set((args.data_dir / "input.txt").read_text()))
    initial = _initial(vocab)
    p = {n: mx.array(w, dtype=mx.float32) for n, w in initial.items()}
    m, v = [{n: mx.zeros_like(w) for n, w in p.items()} for _ in range(2)]
    variants = (
        ["mlx_manual", "mlx_fast"]
        if args.backend == "mlx_pair"
        else [args.backend]
    )
    states = {name: (p, m, v) for name in variants}
    functions = {name: _step_function(name == "mlx_fast") for name in variants}
    rng = np.random.default_rng(1337)

    def consume_evaluation():
        for values in [train, val]:
            for _ in range(20):
                rng.integers(len(values) - 256, size=12)

    consume_evaluation()
    rows = []
    for iteration in range(args.steps):
        if iteration and iteration % 200 == 0:
            consume_evaluation()
        input_started = time.perf_counter()
        offsets = rng.integers(len(train) - 256, size=12)
        tok = mx.array(
            np.stack([train[i : i + 256] for i in offsets]).astype(np.int64)
        )
        nxt = mx.array(
            np.stack([train[i + 1 : i + 257] for i in offsets]).astype(np.int64)
        )
        lr = (
            0.001 * (iteration + 1) / 101
            if iteration < 100
            else 0.0001
            + 0.5
            * (0.001 - 0.0001)
            * (1 + np.cos(np.pi * (iteration - 100) / 1900))
        )
        scalars = [
            mx.array(np.float32(x))
            for x in [
                lr,
                1 - 0.9 ** (iteration + 1),
                1 - 0.99 ** (iteration + 1),
            ]
        ]
        mx.eval(tok, nxt, scalars)
        input_seconds = time.perf_counter() - input_started
        order = variants if iteration % 2 == 0 else variants[::-1]
        for name in order:
            p, m, v = states[name]
            started = time.perf_counter()
            p, m, v, loss, status = functions[name](p, m, v, tok, nxt, *scalars)
            mx.eval(p, m, v, loss, status)
            if not bool(status.item()):
                raise ValueError("nonfinite control output")
            loss = float(loss.item())
            seconds = time.perf_counter() - started
            states[name] = (p, m, v)
            row = {
                "variant": name,
                "iteration": iteration,
                "loss": loss,
                "seconds": seconds,
                "input_seconds": input_seconds,
                "iteration_seconds": seconds + input_seconds,
                "lr": lr,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for name, (p, m, v) in states.items():
        checkpoint = args.out_dir / (
            f"{name}-last.npz" if args.backend == "mlx_pair" else "last.npz"
        )
        np.savez(
            checkpoint,
            **{
                n + "_" + suffix: np.asarray(t[n])
                for suffix, t in [("w", p), ("m", m), ("v", v)]
                for n in p
            },
        )
        checkpoints[name] = str(checkpoint)
    result = {
        "backend": args.backend,
        "parameters": sum(w.size for w in initial.values()),
        "dropout": 0,
        "mlx_version": importlib.metadata.version("mlx"),
        "dtype": "float32, TF32 disabled",
        "device": str(mx.default_device()),
        "train_sha256": hashlib.sha256(
            (args.data_dir / "train.bin").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "scope": "Complete no-dropout model forward, MLX-autodiff backward, clipping, AdamW and numerical-finiteness checks. Independent control, not SQL execution. Input preparation/evaluation/I/O outside timer.",
        "rows": rows,
        "warm_seconds": {
            name: statistics.mean(
                r["seconds"]
                for r in rows
                if r["variant"] == name and r["iteration"] > 0
            )
            for name in variants
        },
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
