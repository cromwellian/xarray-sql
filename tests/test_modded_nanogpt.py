"""Historical speedrun data, optimizer and SQL training contracts."""

import importlib
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def recipe(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "benchmarks/ml"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("_modded_nanogpt")


def _shard(path, tokens, *, magic=20240520, version=1, count=None):
    header = np.zeros(256, dtype="<i4")
    header[:3] = magic, version, len(tokens) if count is None else count
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype="<u2").tofile(handle)


def test_historical_model_and_unshortened_schedule(recipe):
    config = recipe._Config()
    assert config.manifest()["parameters"] == 123568128
    assert config.manifest()["tokens_per_update"] == 1024
    assert config.lr(0) == config.lr(3649) == config.lr(3650) == 0.0036
    assert config.lr(5099) == pytest.approx(0.0036 / 1450)
    with pytest.raises(ValueError, match="outside"):
        config.lr(5100)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 7},
        {"batch": 0},
        {"accumulation": 0},
        {"ns_steps": 0},
        {"warmdown": 0},
        {"learning_rate": float("nan")},
    ],
)
def test_reject_invalid_recipes(recipe, kwargs):
    with pytest.raises(ValueError):
        recipe._Config(**kwargs)


def test_fineweb_sequential_shift_shards_and_reset(recipe, tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    _shard(a, np.arange(14))
    _shard(b, np.arange(20, 34))
    stream = recipe._ShardStream([b, a], recipe._Config(batch=2, context=3))
    expected = [
        np.arange(7),
        np.arange(6, 13),
        np.arange(20, 27),
        np.arange(26, 33),
        np.arange(7),
    ]
    for values in expected:
        tok, nxt = stream.next()
        np.testing.assert_array_equal(tok, values[:-1].reshape(2, 3))
        np.testing.assert_array_equal(nxt, values[1:].reshape(2, 3))
    stream.reset()
    np.testing.assert_array_equal(stream.next()[0], np.arange(6).reshape(2, 3))


@pytest.mark.parametrize(
    "options", [{"magic": 0}, {"version": 2}, {"count": 999}]
)
def test_fineweb_rejects_header_and_size_mismatch(recipe, tmp_path, options):
    path = tmp_path / "bad.bin"
    _shard(path, np.arange(20), **options)
    with pytest.raises(ValueError, match="FineWeb"):
        recipe._ShardStream([path], recipe._Config(batch=1, context=4))


def test_fineweb_rejects_out_of_vocabulary_tokens(recipe, tmp_path):
    path = tmp_path / "bad.bin"
    _shard(path, [0, 50257, 1, 2, 3])
    stream = recipe._ShardStream([path], recipe._Config(batch=1, context=4))
    with pytest.raises(ValueError, match="vocabulary"):
        stream.next()


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("gpu", [False, True])
def test_sql_matches_pinned_upstream_gradients_and_optimizer(
    recipe, accumulation, gpu
):
    pytest.importorskip("ddxdb")
    pytest.importorskip("einfold.dense_program")
    pytest.importorskip("torch")
    if gpu and os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("MLX GPU execution requires MLX_ENABLE_TF32=0")
    upstream = Path(
        os.environ.get(
            "MODDED_NANOGPT_SOURCE",
            Path(__file__).resolve().parents[2] / "modded-nanogpt",
        )
    )
    if not (upstream / recipe._SOURCE).exists():
        pytest.skip("requires the pinned Modded-NanoGPT checkout")
    sql = importlib.import_module("_modded_sql")
    torch = importlib.import_module("_modded_torch")
    config = recipe._Config(
        layers=2,
        heads=2,
        width=8,
        vocab=24,
        context=4,
        batch=2,
        accumulation=accumulation,
        seed=17,
    )
    params = recipe._initial(config)
    actual = sql._SQLTrainer(config, params, gpu=gpu, compiled=gpu, verify=True)
    expected = torch._TorchTrainer(config, params, upstream, "cpu")
    rng = np.random.default_rng(18)
    try:
        for iteration in range(5):
            actual.schedule(iteration)
            expected.schedule(iteration)
            for micro in range(accumulation):
                pair = [
                    rng.integers(
                        config.vocab,
                        size=(config.batch, config.context),
                        dtype=np.int64,
                    )
                    for _ in range(2)
                ]
                final = micro == accumulation - 1
                result = actual.microstep(*pair, final=final)
                loss = expected.microstep(*pair, final=final)
                np.testing.assert_allclose(
                    result.loss_value(), loss, atol=3e-5, rtol=3e-5
                )
            arrays = actual.arrays(result, gradients=True)
            reference = expected.arrays(gradients=True)
            assert arrays.keys() == reference.keys()
            for name in arrays:
                np.testing.assert_allclose(
                    arrays[name],
                    reference[name],
                    atol=3e-5,
                    rtol=3e-5,
                    err_msg=f"update {iteration}: {name}",
                )
        # A separate loss query must read current weights and the new batch.
        pair = [
            rng.integers(
                config.vocab,
                size=(config.batch, config.context),
                dtype=np.int64,
            )
            for _ in range(2)
        ]
        np.testing.assert_allclose(
            actual.evaluate(*pair),
            expected.evaluate(*pair),
            atol=3e-5,
            rtol=3e-5,
        )
    finally:
        actual.close()
