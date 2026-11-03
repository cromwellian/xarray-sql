"""Owned SQL/ddx execution for the pinned Modded-NanoGPT recipe."""

from __future__ import annotations

import numpy as np
import xarray as xr
from _hosts import DenseCPU, DenseGPU
from _modded_nanogpt import (
    _LOSS,
    _adam,
    _batch,
    _dims,
    _gradients,
    _model,
    _muon,
    _static,
    _table,
)

import xarray_sql as xql
from xarray_sql.training import TrainingSession


class _SQLTrainer:
    def __init__(
        self,
        config,
        params,
        *,
        gpu=True,
        compiled=True,
        verify=False,
        validation="full",
        cache_gib=16,
    ):
        self.config = config
        self.host = DenseGPU() if gpu else DenseCPU()
        self.session = TrainingSession(
            self.host.ctx,
            precision=self.host.dense_precision,
            max_cached_bytes=int(cache_gib * (1 << 30)),
            inline_ctes=self.host.executor.inline_ctes,
        )
        for name, value in params.items():
            self._load(name, _table(name, w=value))
            state = {"m": np.zeros_like(value)}
            if name == "wte":
                state["v"] = np.zeros_like(value)
            self._load(name + "_state", _table(name, **state))
            if config.accumulation > 1:
                self._load(
                    name + "_accum", _table(name, g=np.zeros_like(value))
                )
        for name, dataset in _static(config).items():
            # Structural coordinate maps stay host constants, not dynamic inputs.
            self.host.load(name, dataset)
        empty = np.zeros((config.batch, config.context), dtype=np.int64)
        self._load("batch", _batch(empty, empty))
        if compiled and gpu:
            self.session.index_domains(
                "batch", {"tok": ("wte", "tok"), "next": ("wte", "tok")}
            )
            self.session.bind("batch")
        self._load("train_step", self._schedule(0))
        self.session.executor.register_view("logits", _model(config))
        gradients = _gradients(config)
        stages = {}
        if config.accumulation > 1:
            for name in params:
                keys = _dims(name)
                same = " AND ".join(f"g.{k}=a.{k}" for k in keys)
                fields = ",".join(f"g.{k}" for k in keys)
                stages[name + "_total"] = (
                    f"SELECT {fields},a.g+g.g/{config.accumulation} AS g FROM {name}_grad g JOIN {name}_accum a ON {same} ORDER BY {fields}"
                )
        accumulation_stages = dict(stages)
        assignments, updates = {}, {}
        for name, shape in config.shapes.items():
            grad = name + ("_total" if accumulation_stages else "_grad")
            if name == "wte":
                updates[name + "_next"] = _adam(grad)
            else:
                numerical_stages, update = _muon(
                    name, shape, config.ns_steps, grad
                )
                stages.update(numerical_stages)
                updates[name + "_next"] = update
            dims = list(_dims(name))
            assignments[name] = (name + "_next", dims + ["w"])
            assignments[name + "_state"] = (
                name + "_next",
                dims + ["m", "v"] if name == "wte" else dims + ["m"],
            )
            if accumulation_stages:
                updates[name + "_zero"] = (
                    f"SELECT {','.join(dims)},0.0 AS g FROM {name}_accum ORDER BY {','.join(dims)}"
                )
                assignments[name + "_accum"] = (name + "_zero", dims + ["g"])
        options = {
            "gradients": gradients,
            "stages": stages,
            "compile": compiled and gpu,
            "capture_intermediates": verify,
            "validation": validation,
            # Two prepared graphs add their structural stage tables on first
            # use; they must observe each other's catalog generations.
            "freeze_context": compiled and gpu and config.accumulation == 1,
        }
        self.final = self.session.prepare_step(
            updates=updates, assignments=assignments, **options
        )
        self.accumulate = None
        if accumulation_stages:
            self.accumulate = self.session.prepare_step(
                updates={},
                assignments={
                    name + "_accum": (name + "_total", [*_dims(name), "g"])
                    for name in params
                },
                **(options | {"stages": accumulation_stages}),
            )

    def _load(self, name, dataset):
        self.host.load(name, dataset)
        self.session.bind(name)

    def _schedule(self, iteration):
        return xr.Dataset(
            {
                "lr": xr.DataArray(self.config.lr(iteration)),
                "b1": xr.DataArray(1 - 0.9 ** (iteration + 1)),
                "b2": xr.DataArray(1 - 0.95 ** (iteration + 1)),
            }
        )

    def schedule(self, iteration):
        self.session.bind("train_step", self._schedule(iteration))

    def microstep(self, tok, nxt, final):
        self.session.bind("batch", _batch(tok, nxt))
        return (self.final if final else self.accumulate)()

    def evaluate(self, tok, nxt):
        self.session.bind("batch", _batch(tok, nxt))
        rows = self.session.query(f"WITH {_LOSS} SELECT l FROM loss")
        return float(rows["l"][0].as_py())

    def arrays(self, result=None, gradients=False):
        specs = {}
        for name in self.config.shapes:
            specs[name] = ("w",)
            specs[name + "_state"] = ("m", "v") if name == "wte" else ("m",)
            if gradients:
                specs[
                    name
                    + ("_total" if self.config.accumulation > 1 else "_grad")
                ] = ("g",)
        current = self.session.export(
            *(name for name in specs if not name.endswith(("_grad", "_total")))
        )
        if gradients:
            current.update(
                result.export(
                    *(
                        name
                        for name in specs
                        if name.endswith(("_grad", "_total"))
                    )
                )
            )
        arrays = {}
        for table, fields in specs.items():
            name = (
                table.removesuffix("_state")
                .removesuffix("_grad")
                .removesuffix("_total")
            )
            ds = xql.to_dataset(current[table], dims=list(_dims(name)))
            for field in fields:
                arrays[name + "_" + field] = ds[field].values
        return arrays

    def close(self):
        self.session.close()
