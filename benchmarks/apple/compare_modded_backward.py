"""Compare exact historical model loss/gradients without optimizer work.

The Modded-NanoGPT recipe (https://github.com/kellerjordan/modded-nanogpt)
uses tied embeddings, rotary Q/K, RMS normalization and squared ReLU.
This independent MLX autodiff control is diagnostic, not a SQL executor.
Weights are held fixed; batches advance. Uploads and exports are outside
timers, which synchronize loss, every gradient, and finite-value checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from xarray_sql.training import TrainingSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from _hosts import DenseGPU
from _modded_nanogpt import (
    _batch,
    _Config,
    _dims,
    _gradients,
    _initial,
    _model,
    _ShardStream,
    _static,
    _table,
)


def _loss(params, tok, nxt, config, fast):
    def norm(x):
        return x / mx.sqrt(
            mx.mean(x * x, axis=-1, keepdims=True) + np.finfo(np.float32).eps
        )

    angles = mx.arange(config.context)[:, None] * (
        1.0
        / (
            10000
            ** (mx.arange(0, config.head, 2, dtype=mx.float32) / config.head)
        )
    )
    co, si = mx.cos(angles)[None, :, None], mx.sin(angles)[None, :, None]

    def rotate(x):
        a, b = mx.split(x, 2, axis=-1)
        return mx.concatenate([a * co + b * si, -a * si + b * co], axis=-1)

    x = params["wte"][tok]
    causal = (
        mx.arange(config.context)[:, None] >= mx.arange(config.context)[None, :]
    )
    for layer in range(config.layers):
        prefix = f"l{layer}_"
        h = norm(x)
        q, k, v = [
            (h @ params[prefix + name].T).reshape(
                config.batch, config.context, config.heads, config.head
            )
            for name in ("wq", "wk", "wv")
        ]
        q, k, v = (
            norm(rotate(q)).swapaxes(1, 2),
            norm(rotate(k)).swapaxes(1, 2),
            v.swapaxes(1, 2),
        )
        if fast:
            heads = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=config.head**-0.5,
                mask="causal",
                force_fused=True,
            )
        else:
            scores = mx.where(
                causal,
                (q @ k.swapaxes(-1, -2)) / config.head**0.5,
                -float("inf"),
            )
            weights = mx.exp(
                scores
                - mx.stop_gradient(mx.max(scores, axis=-1, keepdims=True))
            )
            heads = (weights / mx.sum(weights, axis=-1, keepdims=True)) @ v
        x = x + heads.swapaxes(1, 2).reshape(x.shape) @ params[prefix + "wo"].T
        h = norm(x) @ params[prefix + "fc"].T
        h = mx.maximum(h, 0)
        x = x + (h * h) @ params[prefix + "proj"].T
    logits = norm(x) @ params["wte"].T
    target = mx.take_along_axis(logits, nxt[..., None], axis=-1)[..., 0]
    return mx.mean(mx.logsumexp(logits, axis=-1) - target)


def _native(config, params, fast):
    differentiated = mx.value_and_grad(
        lambda p, t, n: _loss(p, t, n, config, fast)
    )

    def compute(p, tok, nxt):
        loss, grads = differentiated(p, tok, nxt)
        status = mx.isfinite(loss)
        for value in grads.values():
            status = status & mx.all(mx.isfinite(value))
        return loss, grads, status

    return mx.compile(compute), {k: mx.array(v) for k, v in params.items()}


def _sql(config, params):
    host = DenseGPU()
    session = TrainingSession(
        host.ctx,
        precision="float32",
        max_cached_bytes=16 << 30,
        inline_ctes=host.executor.inline_ctes,
    )
    for name, value in params.items():
        host.load(name, _table(name, w=value))
        session.bind(name)
    for name, dataset in _static(config).items():
        host.load(name, dataset)
    empty = np.zeros((config.batch, config.context), dtype=np.int64)
    host.load("batch", _batch(empty, empty))
    session.bind("batch")
    session.index_domains(
        "batch", {"tok": ("wte", "tok"), "next": ("wte", "tok")}
    )
    session.executor.register_view("logits", _model(config))
    step = session.prepare_step(
        gradients=_gradients(config),
        stages={},
        updates={},
        assignments={},
        compile=True,
        capture_intermediates=True,
        validation="full",
        freeze_context=True,
    )
    return session, step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "sql"], required=True)
    parser.add_argument("--attention", choices=["math", "fast"], default="fast")
    parser.add_argument("--preset", choices=["124m", "tiny"], default="124m")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradients", type=Path)
    args = parser.parse_args()
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        parser.error("requires MLX_ENABLE_TF32=0")
    if not 0 <= args.warmup < args.steps:
        parser.error("require 0 <= warmup < steps")
    if args.output.exists() or (args.gradients and args.gradients.exists()):
        parser.error("output paths must be new")
    config = (
        _Config()
        if args.preset == "124m"
        else _Config(layers=2, heads=2, width=128, vocab=32, context=32)
    )
    params = _initial(config)
    if args.weights:
        with np.load(args.weights) as saved:
            params = {name: saved[name + "_w"].copy() for name in config.shapes}
    for name, shape in config.shapes.items():
        if (
            params[name].shape != shape
            or params[name].dtype != np.float32
            or not np.isfinite(params[name]).all()
        ):
            parser.error(
                f"invalid float32 weights for {name}: expected {shape}"
            )
    weight_hashes = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in params.items()
    }
    setup = time.perf_counter()
    session = None
    if args.backend == "native":
        compute, weights = _native(config, params, args.attention == "fast")
        mx.eval(weights)
    else:
        session, compute = _sql(config, params)
    setup = time.perf_counter() - setup
    stream = _ShardStream([args.data], config)
    samples, losses = [], []
    try:
        for _ in range(args.steps):
            tok, nxt = stream.next()
            if args.backend == "native":
                inputs = mx.array(tok), mx.array(nxt)
                mx.eval(inputs)
            else:
                session.bind("batch", _batch(tok, nxt))
            start = time.perf_counter()
            if args.backend == "native":
                loss, gradients, status = compute(weights, *inputs)
                mx.eval(loss, gradients, status)
                if not bool(status.item()):
                    raise ValueError("nonfinite native gradient or loss")
                loss = float(loss.item())
            else:
                result = compute()
                loss = float(result.loss)
            samples.append(time.perf_counter() - start)
            losses.append(loss)
            print(
                json.dumps(
                    {"step": len(samples), "seconds": samples[-1], "loss": loss}
                ),
                flush=True,
            )
        if args.gradients:
            if args.backend == "sql":
                gradients = {}
                for name, output in result._outputs.items():
                    if not name.endswith("_grad"):
                        continue
                    name = name.removesuffix("_grad")
                    tensor = output.table.resolve(None, "g").tensor
                    axes = tuple(
                        output.table.resolve(None, dim).tensor.axis
                        for dim in _dims(name)
                    )
                    value = np.asarray(tensor.get())
                    value = value.transpose(
                        tuple(
                            tensor.axes.index(a)
                            for a in axes
                            if a in tensor.axes
                        )
                    )
                    gradients[name] = np.broadcast_to(
                        value.reshape(
                            tuple(
                                len(a) if a in tensor.axes else 1 for a in axes
                            )
                        ),
                        config.shapes[name],
                    )
            np.savez(
                args.gradients,
                loss=loss,
                **{k: np.asarray(v) for k, v in gradients.items()},
            )
        args.output.write_text(
            json.dumps(
                {
                    **config.manifest(),
                    "backend": args.backend,
                    "attention": args.attention
                    if args.backend == "native"
                    else "SQL-selected",
                    "workload": "fixed weights, advancing batches, loss and all raw gradients; no optimizer",
                    "mlx": importlib.metadata.version("mlx"),
                    "weight_sha256": weight_hashes,
                    "data_path": str(args.data.resolve()),
                    "symbolic_masks": os.environ.get(
                        "EINFOLD_DENSE_SYMBOLIC_MASKS", "0"
                    ),
                    "setup_seconds": setup,
                    "seconds": samples,
                    "losses": losses,
                    "warmup": args.warmup,
                    "warm_seconds": statistics.mean(samples[args.warmup :]),
                    "peak_active_bytes": mx.get_peak_memory(),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    main()
