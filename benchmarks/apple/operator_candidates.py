"""Measure candidate numerical kernels independently of SQL lowering.

These microbenchmarks do not establish end-to-end training speedups.
Attention has no probability dropout; the training recipe requires it.
"""

import json
import statistics
import time

import mlx.core as mx
import numpy as np


def _time(function, *args):
    for _ in range(3):
        mx.eval(function(*args))
    times = []
    for _ in range(20):
        start = time.perf_counter()
        mx.eval(function(*args))
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    mx.random.seed(1337)
    x = mx.random.normal((3072, 384))
    w = mx.ones((384,))

    def norm(x, w):
        centered = x - mx.mean(x, axis=-1, keepdims=True)
        return (
            centered
            * mx.rsqrt(
                mx.mean(centered * centered, axis=-1, keepdims=True) + 1e-5
            )
            * w
        )

    reference = mx.compile(norm)
    fast = lambda x, w: mx.fast.layer_norm(x, w, None, 1e-5)
    np.testing.assert_allclose(
        np.asarray(reference(x, w)),
        np.asarray(fast(x, w)),
        atol=3e-5,
        rtol=3e-5,
    )
    result = {
        "layer_norm": {
            "compiled_seconds": _time(reference, x, w),
            "fast_seconds": _time(fast, x, w),
        }
    }
    q, k, v = [mx.random.normal((12, 6, 256, 64)) for _ in range(3)]
    mask = mx.arange(256)[:, None] >= mx.arange(256)[None, :]

    def attention(q, k, v):
        scores = (q @ k.swapaxes(-1, -2)) * 0.125
        scores = mx.where(mask, scores, -float("inf"))
        e = mx.exp(scores - mx.max(scores, axis=-1, keepdims=True))
        return (e / mx.sum(e, axis=-1, keepdims=True)) @ v

    reference = mx.compile(attention)
    fast = lambda q, k, v: mx.fast.scaled_dot_product_attention(
        q, k, v, scale=0.125, mask="causal"
    )
    np.testing.assert_allclose(
        np.asarray(reference(q, k, v)),
        np.asarray(fast(q, k, v)),
        atol=3e-5,
        rtol=3e-5,
    )
    result["causal_attention_without_dropout"] = {
        "compiled_seconds": _time(reference, q, k, v),
        "fast_seconds": _time(fast, q, k, v),
    }
    a, b = mx.random.normal((3072, 384)), mx.random.normal((384, 1536))
    result["matmul"] = {}
    baseline = a @ b
    for name, dtype in [("float32", mx.float32), ("bfloat16", mx.bfloat16)]:
        left, right = a.astype(dtype), b.astype(dtype)
        result["matmul"][name] = {
            "seconds": _time(lambda a, b: a @ b, left, right),
            "relative_l2_error": float(
                mx.linalg.norm((left @ right).astype(mx.float32) - baseline)
                / mx.linalg.norm(baseline)
            ),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
