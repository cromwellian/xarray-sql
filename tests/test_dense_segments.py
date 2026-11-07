"""Dynamic grouped sums preserve SQL groups and repeat their GPU results."""

import os

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context
from xarray_sql.training import TrainingSession

pytest.importorskip("einfold.dense_program")

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") != "0", reason="requires MLX GPU"
)


@pytest.mark.parametrize("length", [1, 19])
def test_fixed_segments_keep_per_row_keys_and_null_groups(monkeypatch, length):
    monkeypatch.setenv("XQL_MLX_FIXED_SEGMENTS", "1")
    labels = np.array([10, 20, 40, 80])
    rng = np.random.default_rng(23)
    keys = rng.integers(len(labels), size=(3, length))
    values = rng.integers(-4, 6, size=(3, length)).astype(np.float64)
    values[0, 0] = -0.0
    session = TrainingSession(Context())
    try:
        session.bind(
            "categories",
            xr.Dataset({"v": ("label", np.zeros(4))}, coords={"label": labels}),
        )
        session.index_domains("batch", {"label": ("categories", "label")})
        session.bind(
            "batch",
            xr.Dataset(
                {
                    "label": (("r", "pos"), labels[keys]),
                    "v": (("r", "pos"), values),
                }
            ),
        )
        query = """SELECT r,label,SUM(CASE WHEN v>=0 THEN v END) AS total
        FROM batch GROUP BY r,label ORDER BY r,label"""
        expected = {"r": [], "label": [], "total": []}
        for row in range(3):
            for key in sorted(set(keys[row])):
                selected = values[row, keys[row] == key]
                valid = selected[selected >= 0]
                expected["r"].append(row)
                expected["label"].append(int(labels[key]))
                expected["total"].append(
                    float(valid.sum()) if valid.size else None
                )
        first = session.query(query)
        assert first.to_pydict() == expected
        assert first.equals(session.query(query))
        assert all(
            not np.signbit(value)
            for value in first["total"].to_pylist()
            if value == 0
        )
        empty = session.query(
            query.replace("FROM batch", "FROM batch WHERE FALSE")
        )
        assert empty.to_pydict() == {"r": [], "label": [], "total": []}
    finally:
        session.close()
