"""Training policies explicitly distinguish intermediate and state checks."""

import os

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context

pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")
from xarray_sql.training import TrainingSession


@pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") != "0", reason="requires GPU"
)
@pytest.mark.parametrize("validation", ["full", "state", "none"])
@pytest.mark.parametrize("bad_state", [False, True])
@pytest.mark.parametrize("compile_step", [False, True])
def test_validation_policy_and_failure_atomicity(
    validation, bad_state, compile_step
):
    ctx = Context()
    ctx.from_dataset("w", xr.Dataset({"v": ("i", [1.0, 2.0])}))
    session = TrainingSession(ctx)
    session.bind("w")
    step = session.prepare_step(
        gradients={
            "g": "WITH loss AS (SELECT SUM(v*v) AS l FROM w) SELECT i, grad(loss,v) AS g FROM w ORDER BY i"
        },
        stages={"unused": "SELECT i,sqrt(-v*v) AS bad FROM w ORDER BY i"},
        updates={
            "u": "SELECT i,"
            + ("sqrt(-v*v)" if bad_state else "v")
            + " AS v FROM w ORDER BY i"
        },
        assignments={"w": ("u", ["i", "v"])},
        compile=compile_step,
        capture_intermediates=False,
        freeze_context=compile_step,
        validation=validation,
    )
    before = session.export("w")["w"]
    if validation == "full" or (validation == "state" and bad_state):
        with pytest.raises(ValueError, match="finite"):
            step()
        assert session.version == 0
        assert session.export("w")["w"].equals(before)
    else:
        result = step()
        assert result.loss_value() == 5.0
        if bad_state:
            # Trusted execution can propagate NaNs into a later device step.
            # Ordinary Arrow export retains its separate finite-value contract.
            assert np.isnan(step().loss_value())
        else:
            values = session.export("w")["w"]["v"].to_numpy()
            np.testing.assert_array_equal(values, [1.0, 2.0])


def test_policy_configuration_rejects_unknown_modes_and_uncompiled_freezing():
    session = TrainingSession(Context(), precision="float64")
    for options in ({"validation": "invalid"}, {"freeze_context": True}):
        with pytest.raises(ValueError):
            session.prepare_step(
                gradients={}, stages={}, updates={}, assignments={}, **options
            )


@pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") != "0", reason="requires GPU"
)
@pytest.mark.parametrize("validation", ["full", "state", "none"])
@pytest.mark.parametrize(
    "query",
    [
        "SELECT i,v FROM w WHERE v < 0 ORDER BY i",
        "SELECT i,CAST(NULL AS DOUBLE) AS v FROM w ORDER BY i",
    ],
)
def test_output_row_and_null_contracts_follow_policy(validation, query):
    ctx = Context()
    ctx.from_dataset("w", xr.Dataset({"v": ("i", [1.0, 2.0])}))
    session = TrainingSession(ctx)
    session.bind("w")
    step = session.prepare_step(
        gradients={
            "g": "WITH loss AS (SELECT SUM(v*v) AS l FROM w) SELECT i,grad(loss,v) AS g FROM w ORDER BY i"
        },
        stages={},
        updates={"u": query},
        assignments={"w": ("u", ["i", "v"])},
        compile=True,
        validation=validation,
        freeze_context=True,
        capture_intermediates=False,
    )
    if validation == "none":
        step()
        assert session.version == 1
    else:
        with pytest.raises(ValueError, match="rectangular|nonnullable|finite"):
            step()
        assert session.version == 0
