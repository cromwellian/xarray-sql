"""GPU bag accumulation preserves ordered, permuted and partial domains."""

import os

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context

pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")

from xarray_sql.dense import DenseExecutor

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") != "0",
    reason="GPU requires MLX_ENABLE_TF32=0",
)


@pytest.mark.parametrize("direct", ["0", "1"])
@pytest.mark.parametrize(
    "coordinates",
    [
        ([0, 2, 5], [0, 2, 5]),
        ([0, 2, 5], [5, 0, 2]),
        ([0, 2, 5], [2, 7]),
        ([5, 0, 2], [3, 1, 2]),
    ],
)
def test_gpu_union_keeps_nulls_duplicates_and_coordinate_mappings(
    monkeypatch, direct, coordinates
):
    monkeypatch.setenv("XQL_MLX_DIRECT_UNION", direct)
    ctx = Context()
    sources = {}
    for name, labels in zip(["a", "b"], coordinates):
        values = np.arange(len(labels) * 2, dtype=float).reshape(-1, 2) - 2
        sources[name] = xr.Dataset(
            {"v": (("i", "j"), values)}, coords={"i": labels, "j": [0, 1]}
        )
        ctx.from_dataset(name, sources[name])
    executor = DenseExecutor(ctx, precision="float32", fallback=False)
    sql = """WITH u AS (
      SELECT i,j,CASE WHEN v>0 THEN v END AS v FROM a
      UNION ALL SELECT i,j,CASE WHEN v>0 THEN v END AS v FROM b
      UNION ALL SELECT i,j,CASE WHEN v>0 THEN v END AS v FROM a)
      SELECT i,j,SUM(v) AS s FROM u GROUP BY i,j ORDER BY i,j"""
    run = executor.prepare(sql)
    first = run()
    assert first.to_pydict() == ctx.sql(sql).to_arrow_table().to_pydict()
    saved = first.to_pydict()
    current = sources["a"].copy(deep=True)
    current["v"] += 1
    second = run(a=current)
    assert second.to_pydict() == ctx.sql(sql).to_arrow_table().to_pydict()
    assert first.to_pydict() == saved
