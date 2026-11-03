"""Numerical attention lowering preserves SQL domains, NULLs and derivatives."""

import os

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context

pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")

from einfold.dense_program import _Decline

from xarray_sql.dense import DenseExecutor

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") != "0",
    reason="GPU tests require MLX_ENABLE_TF32=0",
)


def _query(case):
    predicate = {
        "causal": "q.i>=k.j",
        "irregular": "q.i>=k.j AND k.j<>2",
        "empty": "q.i<k.j",
        "dynamic": "q.i>=k.j AND q.val>0",
    }.get(case, "q.i>=k.j")
    values = (
        "CASE WHEN j=2 THEN NULL ELSE val END" if case == "nullable" else "val"
    )
    extra = " * d.scale" if case == "dropout" else ""
    join = (
        "JOIN d ON a.b=d.b AND a.h=d.h AND a.i=d.i AND a.j=d.j" if extra else ""
    )
    maximum_filter = "WHERE j<>2" if case == "different_maximum" else ""
    value_filter = "AND v.c=a.i" if case == "shared_value_axis" else ""
    divisor = {"huge_divisor": "1e300", "tiny_divisor": "1e-300"}.get(
        case, "8.0"
    )
    return f"""WITH values_table AS (SELECT b,h,j,c,{values} AS val FROM v),
      scores AS (SELECT q.b,q.h,q.i,k.j,SUM(q.val*k.val)/{divisor} AS s
        FROM q JOIN k ON q.b=k.b AND q.h=k.h AND q.c=k.c
        WHERE {predicate} GROUP BY q.b,q.h,q.i,k.j),
      maximum AS (SELECT b,h,i,ddx_stop_gradient(MAX(s)) AS m FROM scores {maximum_filter} GROUP BY b,h,i),
      weights AS (SELECT s.b,s.h,s.i,s.j,exp(s.s-m.m) AS p FROM scores s JOIN maximum m
        ON s.b=m.b AND s.h=m.h AND s.i=m.i),
      denominator AS (SELECT b,h,i,SUM(p) AS z FROM weights GROUP BY b,h,i),
      attention AS (SELECT a.b,a.h,a.i,v.c,SUM(a.p/n.z{extra}*v.val) AS val
        FROM weights a JOIN denominator n ON a.b=n.b AND a.h=n.h AND a.i=n.i
        JOIN values_table v ON a.b=v.b AND a.h=v.h AND a.j=v.j {value_filter} {join} GROUP BY a.b,a.h,a.i,v.c),
      loss AS (SELECT SUM(val*val) AS l FROM attention)
    """


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("length", [5, 32])
@pytest.mark.parametrize(
    "case",
    [
        "causal",
        "irregular",
        "empty",
        "dynamic",
        "nullable",
        "dropout",
        "different_maximum",
        "shared_value_axis",
        "huge_divisor",
        "tiny_divisor",
    ],
)
def test_attention_and_ddx_gradients_keep_sql_semantics(
    monkeypatch, seed, case, length
):
    rng = np.random.default_rng(seed)
    ctx = Context()
    coordinates = np.arange(length) * 2 - 4 if seed % 2 else np.arange(length)
    dims = {
        "q": ("b", "h", "i", "c"),
        "k": ("b", "h", "j", "c"),
        "v": ("b", "h", "j", "c"),
    }
    datasets = {}
    for name, axes in dims.items():
        datasets[name] = xr.Dataset(
            {"val": (axes, rng.normal(size=(1, 2, length, 64)))},
            coords={axes[2]: coordinates},
        )
        ctx.from_dataset(name, datasets[name])
    ctx.from_dataset(
        "d",
        xr.Dataset(
            {
                "scale": (
                    ("b", "h", "i", "j"),
                    2.0 * (rng.random((1, 2, length, length)) > 0.5),
                )
            },
            coords={"i": coordinates, "j": coordinates},
        ),
    )
    root = _query(case)
    statements = [root + "SELECT b,h,c,i,val FROM attention ORDER BY b,h,c,i"]
    for name, axes in dims.items():
        columns = ",".join(axes)
        statements.append(
            root
            + f"SELECT {columns},grad(loss,val) AS val FROM {name} ORDER BY {columns}"
        )
    monkeypatch.setenv("XQL_MLX_ATTENTION", "0")
    reference = DenseExecutor(ctx, precision="float32", fallback=False)
    monkeypatch.setenv("XQL_MLX_ATTENTION", "1")
    candidate = DenseExecutor(ctx, precision="float32", fallback=False)
    # Reuse each prepared query after changing input values; compiled kernels
    # must not capture the first set of weights.
    for query in statements:
        expected, actual = reference.prepare(query), candidate.prepare(query)
        for _ in range(2):
            try:
                e = expected()
            except _Decline:
                with pytest.raises(_Decline):
                    actual()
                continue
            a = actual()
            assert a.schema == e.schema
            assert a.num_rows == e.num_rows
            for ac, ec in zip(a.columns, e.columns):
                assert ac.is_null().to_pylist() == ec.is_null().to_pylist()
                np.testing.assert_allclose(
                    ac.to_numpy(),
                    ec.to_numpy(),
                    atol=3e-5,
                    rtol=3e-5,
                    equal_nan=True,
                )
            datasets["q"]["val"].values[:] *= 0.99


@pytest.mark.parametrize("key_length", [17, 48])
def test_rectangular_attention_keeps_unused_keys_and_all_gradients(
    monkeypatch, key_length
):
    rng = np.random.default_rng(key_length)
    ctx = Context()
    dims = {
        "q": ("b", "h", "i", "c"),
        "k": ("b", "h", "j", "c"),
        "v": ("b", "h", "j", "c"),
    }
    for name, axes in dims.items():
        length = 32 if name == "q" else key_length
        ctx.from_dataset(
            name,
            xr.Dataset({"val": (axes, rng.normal(size=(1, 2, length, 64)))}),
        )
    monkeypatch.setenv("XQL_MLX_ATTENTION", "0")
    reference = DenseExecutor(ctx, precision="float32", fallback=False)
    monkeypatch.setenv("XQL_MLX_ATTENTION", "1")
    candidate = DenseExecutor(ctx, precision="float32", fallback=False)
    root = _query("causal")
    statements = [root + "SELECT b,h,i,c,val FROM attention ORDER BY b,h,i,c"]
    statements += [
        root
        + f"SELECT {','.join(axes)},grad(loss,val) AS val FROM {name} ORDER BY {','.join(axes)}"
        for name, axes in dims.items()
    ]
    for sql in statements:
        try:
            expected = reference.prepare(sql)()
        except _Decline:
            with pytest.raises(_Decline):
                candidate.prepare(sql)()
            continue
        actual = candidate.prepare(sql)()
        assert actual.schema == expected.schema
        assert actual.num_rows == expected.num_rows
        for a, e in zip(actual.columns, expected.columns):
            assert a.is_null().to_pylist() == e.is_null().to_pylist()
            np.testing.assert_allclose(
                a.to_numpy(), e.to_numpy(), rtol=3e-5, atol=3e-5, equal_nan=True
            )
