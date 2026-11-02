"""Matched JAX training control for the SQL Shakespeare recipe.

JAX (https://github.com/jax-ml/jax) runs through the MLX-backed MetalJAX
plugin (https://github.com/eterevsky/metaljax). Run in an isolated environment
with JAX_PLATFORMS=metal. CPU is an explicit numerical control, never fallback.
Dropout masks and batches use the SQL recipe's NumPy streams; sampling and
device upload are outside the synchronized full-training-step timer.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
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
    exec(  # noqa: S102 -- Only initialization functions from trusted repository source.
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
        scope,
    )
    return scope["init_params"](vocab, np.random.default_rng(1337))


def _masks(rng):
    shapes = {"embed_drop": (12, 256, 384)}
    for layer in range(6):
        shapes[f"l{layer}_adrop"] = (12, 6, 256, 256)
        for name in ("rdrop", "mdrop"):
            shapes[f"l{layer}_{name}"] = (12, 256, 384)
    return {
        n: ((rng.random(s) >= 0.2).astype(np.float64) / 0.8).astype(np.float32)
        for n, s in shapes.items()
    }


def _norm(x, gain):
    centered = x - x.mean(-1, keepdims=True)
    return (
        centered
        / jnp.sqrt((centered * centered).mean(-1, keepdims=True) + 1e-5)
        * gain
    )


def _loss(p, tok, nxt, masks):
    x = (p["wte"][tok] + p["wpe"][None]) * masks["embed_drop"]
    causal = jnp.arange(256)[:, None] >= jnp.arange(256)[None, :]
    for layer in range(6):
        h = _norm(x, p["ln_1"][layer])
        q, k, v = (
            jnp.einsum("bte,ehc->bhtc", h, p[w][layer])
            for w in ("wq", "wk", "wv")
        )
        scores = jnp.einsum("bhic,bhjc->bhij", q, k) / 8
        a = (
            jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), -1)
            * masks[f"l{layer}_adrop"]
        )
        heads = jnp.einsum("bhij,bhjc->bhic", a, v)
        x = (
            x
            + jnp.einsum("bhtc,hce->bte", heads, p["wo"][layer])
            * masks[f"l{layer}_rdrop"]
        )
        h = _norm(x, p["ln_2"][layer])
        hidden = jax.nn.gelu(h @ p["w_fc"][layer], approximate=False)
        x = x + (hidden @ p["w_proj"][layer]) * masks[f"l{layer}_mdrop"]
    logits = _norm(x, p["ln_f"]) @ p["wte"].T
    target = jnp.take_along_axis(logits, nxt[..., None], -1)[..., 0]
    return jnp.mean(jax.nn.logsumexp(logits, -1) - target)


def _step(p, m, v, tok, nxt, masks, lr, b1, b2):
    loss, grads = jax.value_and_grad(_loss)(p, tok, nxt, masks)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in grads.values()))
    clip = jnp.minimum(1.0, 1.0 / (norm + 1e-6))
    new, nm, nv = {}, {}, {}
    for name, w in p.items():
        g = grads[name] * clip
        nm[name] = 0.9 * m[name] + 0.1 * g
        nv[name] = 0.99 * v[name] + 0.01 * g * g
        decay = 0.0 if name.startswith("ln_") else 0.1
        new[name] = w * (1 - lr * decay) - lr * (nm[name] / b1) / (
            jnp.sqrt(nv[name] / b2) + 1e-8
        )
    return new, nm, nv, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--platform", choices=["metal", "cpu"], default="metal")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "nanoGPT/data/shakespeare_char",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.steps <= 2001:
        parser.error("--steps must be between 2 and 2001 for this recipe")
    jax.config.update("jax_enable_x64", False)
    devices = jax.devices()
    if any(d.platform != args.platform for d in devices):
        raise RuntimeError(
            f"expected only {args.platform} devices, got {devices}"
        )
    print("DEVICES", devices, flush=True)
    train = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")
    vocab = len(set((args.data_dir / "input.txt").read_text()))
    initial = _initial(vocab)
    p = {n: jnp.asarray(w, dtype=jnp.float32) for n, w in initial.items()}
    m, v = ({n: jnp.zeros_like(w) for n, w in p.items()} for _ in range(2))
    data_rng, mask_rng = (
        np.random.default_rng(1337),
        np.random.default_rng(1338),
    )
    # SQL's iteration-zero evaluation consumes 20 batches in each split.
    for values in (train, val):
        for _ in range(20):
            data_rng.integers(len(values) - 256, size=12)
    step = jax.jit(_step)
    jax.block_until_ready((p, m, v))
    rows = []
    for iteration in range(args.steps):
        input_started = time.perf_counter()
        starts = data_rng.integers(len(train) - 256, size=12)
        tok = np.stack([train[i : i + 256] for i in starts]).astype(np.int32)
        nxt = np.stack([train[i + 1 : i + 257] for i in starts]).astype(
            np.int32
        )
        inputs = jax.device_put((tok, nxt, _masks(mask_rng)))
        lr = (
            0.001 * (iteration + 1) / 101
            if iteration < 100
            else 0.0001
            + 0.5
            * (0.001 - 0.0001)
            * (1 + np.cos(np.pi * (iteration - 100) / 1900))
        )
        scalars = jax.device_put(
            tuple(
                np.float32(x)
                for x in (
                    lr,
                    1 - 0.9 ** (iteration + 1),
                    1 - 0.99 ** (iteration + 1),
                )
            )
        )
        jax.block_until_ready((inputs, scalars))
        start = time.perf_counter()
        p, m, v, loss = step(p, m, v, *inputs, *scalars)
        jax.block_until_ready((p, m, v, loss))
        loss = float(loss)
        seconds = time.perf_counter() - start
        if not np.isfinite(loss):
            raise ValueError("nonfinite training loss")
        row = {
            "iteration": iteration,
            "seconds": seconds,
            "input_seconds": start - input_started,
            "iteration_seconds": seconds + start - input_started,
            "loss": loss,
            "lr": lr,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "matmul_precision": os.environ.get(
            "METALJAX_MATMUL_PRECISION", "plugin default"
        ),
        "devices": [str(d) for d in devices],
        "platform": args.platform,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "metaljax", "numpy")
        },
        "parameters": sum(w.size for w in initial.values()),
        "dtype": "float32",
        "train_sha256": hashlib.sha256(
            (args.data_dir / "train.bin").read_bytes()
        ).hexdigest(),
        "scope": "Complete forward/backward, clipping and AdamW; input sampling/upload outside timer. Explicit NumPy masks match SQL CPU-mask mode. No extra SQL intermediate validation.",
        "warm_seconds": statistics.mean(r["seconds"] for r in rows[1:]),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(
        args.checkpoint,
        **{
            n + "_" + c: np.asarray(t[n])
            for c, t in [("w", p), ("m", m), ("v", v)]
            for n in p
        },
    )


if __name__ == "__main__":
    main()
