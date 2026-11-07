"""Independent MLX control agrees with the pinned PyTorch model gradients."""

import importlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("fast", [False, True])
def test_native_control_all_parameter_gradients(monkeypatch, fast):
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("requires MLX GPU execution")
    mx = pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks/ml"))
    recipe = importlib.import_module("_modded_nanogpt")
    oracle = importlib.import_module("_modded_torch")
    upstream = root.parent / "modded-nanogpt"
    if not (upstream / recipe._SOURCE).exists():
        pytest.skip("requires pinned upstream model source")
    spec = importlib.util.spec_from_file_location(
        "_native_control", root / "benchmarks/apple/compare_modded_backward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = recipe._Config(
        layers=2, heads=2, width=128, vocab=32, context=32, batch=2
    )
    params = recipe._initial(config)
    rng = np.random.default_rng(19)
    # Exercise attention and hidden gradients beyond zero-output initialization.
    for name, value in params.items():
        if name.endswith(("_wo", "_proj")):
            params[name] = rng.normal(0, 0.02, value.shape).astype(np.float32)
    model = oracle._TorchTrainer(config, params, upstream, "cpu")
    compute, weights = module._native(config, params, fast)
    for _ in range(2):
        tok, nxt = [
            rng.integers(config.vocab, size=(config.batch, config.context))
            for _ in range(2)
        ]
        model.model.zero_grad(set_to_none=True)
        _, expected = model.model(
            torch.tensor(tok), torch.tensor(nxt), return_logits=False
        )
        expected.backward()
        loss, gradients, status = compute(weights, mx.array(tok), mx.array(nxt))
        mx.eval(loss, gradients, status)
        assert bool(status.item())
        np.testing.assert_allclose(
            float(loss.item()), float(expected.detach()), atol=3e-5, rtol=3e-5
        )
        for name, parameter in model.parameters.items():
            np.testing.assert_allclose(
                np.asarray(gradients[name]),
                parameter.grad.numpy(),
                atol=3e-5,
                rtol=3e-5,
            )
