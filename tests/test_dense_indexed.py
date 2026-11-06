"""Compiled indexed partials preserve duplicate ddx contributions."""

import os

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context, to_dataset

pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")

from xarray_sql.training import TrainingSession


@pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") != "0", reason="requires MLX GPU"
)
@pytest.mark.parametrize("indexed", ["0", "1"])
def test_compiled_tied_weight_gradient_combines_duplicate_keys(
    monkeypatch, indexed
):
    monkeypatch.setenv("XQL_MLX_INDEXED_UNION", indexed)
    weights = np.full((16, 3), 0.5)
    coefficients = np.arange(48.0).reshape(16, 3)
    coefficients[3, 0] = 2**24
    session = TrainingSession(Context())
    try:
        session.bind("wte", xr.Dataset({"w": (("tok", "e"), weights)}))
        session.bind("head", xr.Dataset({"a": (("tok", "e"), coefficients)}))
        session.index_domains("batch", {"tok": ("wte", "tok")})
        gradient = """WITH dense AS (
          SELECT SUM(w.w*h.a) AS s FROM wte w JOIN head h ON w.tok=h.tok AND w.e=h.e),
        lookup AS (
          SELECT SUM(CASE WHEN b.scale>0 THEN w.w*b.scale END) AS s
          FROM batch b JOIN wte w ON b.tok=w.tok),
        loss AS (SELECT d.s+b.s AS l FROM dense d CROSS JOIN lookup b)
        SELECT tok,e,grad(loss,w) AS g FROM wte ORDER BY tok,e"""
        step = session.prepare_step(
            gradients={"raw": gradient},
            stages={},
            updates={},
            assignments={},
            compile=True,
        )
        for tokens, scales in (
            ([3, 3, 5, 0], [1.0, 1.0, 2.0, 0.0]),
            ([0, 7, 7, 3], [2.0, 1.0, 3.0, 1.0]),
        ):
            session.bind(
                "batch",
                xr.Dataset(
                    {
                        "tok": (("seq", "pos"), np.array([tokens])),
                        "scale": (("seq", "pos"), np.array([scales])),
                    }
                ),
            )
            result = step()
            rows = result.export("raw")["raw"]
            actual = to_dataset(rows, dims=["tok", "e"])
            expected = coefficients.copy()
            np.add.at(expected, tokens, np.asarray(scales)[:, None])
            np.testing.assert_array_equal(
                actual.g.values, expected.astype(np.float32)
            )
            assert result.loss_value() == pytest.approx(
                float(np.sum(expected * weights))
            )
    finally:
        session.close()
