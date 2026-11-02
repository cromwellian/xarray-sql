"""Predicate proofs retain dynamic GPU NULL masks and fresh input values."""

import os

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context

pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")

from xarray_sql.dense import DenseExecutor


@pytest.mark.parametrize("mode", ["off", "boolean", "presence", "all"])
@pytest.mark.parametrize("width", [0, 3])
@pytest.mark.parametrize(
    "precision",
    [
        "float64",
        pytest.param(
            "float32",
            marks=pytest.mark.skipif(
                os.environ.get("MLX_ENABLE_TF32") != "0",
                reason="GPU requires MLX_ENABLE_TF32=0",
            ),
        ),
    ],
)
def test_predicate_proofs_preserve_dynamic_nulls_and_empty_groups(
    monkeypatch, mode, width, precision
):
    monkeypatch.setenv("EINFOLD_DENSE_PREDICATES", mode)

    def dataset(values):
        return xr.Dataset(
            {"v": ("i", np.asarray(values, dtype=float))},
            coords={"i": [5, -2, 8]},
        )

    ctx = Context()
    ctx.from_dataset("t", dataset([-2, 0, 3]))
    ctx.from_dataset("g", xr.Dataset({"grid": ("j", np.zeros(width))}))
    executor = DenseExecutor(ctx, precision=precision, fallback=False)
    queries = [
        "WITH n AS (SELECT i,j,CASE WHEN v>0 THEN v END AS v FROM t) SELECT i,COUNT(v) AS c,COUNT(v)>0 AS p,CASE WHEN NOT TRUE THEN FALSE ELSE COUNT(v)>0 END AS q,NOT NOT(COUNT(v)>0) AS r FROM n GROUP BY i ORDER BY i",
        "WITH n AS (SELECT i,j,CASE WHEN v>j THEN v END AS v FROM t) SELECT i,COUNT(v) AS c,COUNT(v)>0 AS p FROM n GROUP BY i ORDER BY i",
        "WITH n AS (SELECT i,j,CASE WHEN v>0 THEN v END AS v FROM t),c AS (SELECT i,COUNT(v) AS n FROM n GROUP BY i) SELECT a.i,b.n>0 AS p FROM c a JOIN c b ON a.i=b.i ORDER BY a.i",
    ]
    queries = [
        q.replace("AS v FROM t)", "AS v FROM t CROSS JOIN g)") for q in queries
    ]
    prepared = [executor.prepare(query) for query in queries]
    for values in [[-2, 0, 3], [4, -5, 0]]:
        for query, run in zip(queries, prepared):
            actual = run(t=dataset(values))
            expected = ctx.sql(query).to_arrow_table()
            assert actual.schema == expected.schema
            assert actual.to_pydict() == expected.to_pydict()
