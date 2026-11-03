"""Train a pinned historical Modded-NanoGPT recipe through SQL or PyTorch.

The FineWeb speedrun (https://github.com/KellerJordan/modded-nanogpt)
measures time to validation loss, not five-step throughput. This float32,
single-device port provides both throughput and validation runs. Synthetic
data is explicit and never reported as a time-to-quality measurement.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from _modded_nanogpt import (
    _Config,
    _gradients,
    _initial,
    _model,
    _ShardStream,
)
from _modded_sql import _SQLTrainer

try:
    from _modded_torch import _TorchTrainer
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _TorchTrainer = None


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["sql_mlx", "sql_cpu", "torch_mps", "torch_cpu", "torch_cuda"],
        default="sql_mlx",
    )
    p.add_argument("--preset", choices=["124m", "tiny"], default="124m")
    p.add_argument(
        "--upstream",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "modded-nanogpt",
    )
    p.add_argument("--data-dir", type=Path)
    p.add_argument(
        "--download-data",
        action="store_true",
        help="download train shards and the validation shard (about 200 MB each)",
    )
    p.add_argument("--download-train-shards", type=int, default=1)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="explicit throughput-only random token data",
    )
    p.add_argument(
        "--mode", choices=["throughput", "quality"], default="throughput"
    )
    p.add_argument("--out-dir", type=Path, default=Path("out-modded-sql"))
    p.add_argument(
        "--steps",
        type=int,
        default=20,
        help="exact update cap; does not shorten the 5100-step LR schedule",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="advancing updates excluded only from the warm mean",
    )
    for option in (
        "layers",
        "heads",
        "width",
        "vocab",
        "context",
        "batch",
        "accumulation",
        "iterations",
        "warmdown",
        "ns-steps",
        "seed",
    ):
        p.add_argument("--" + option, type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument(
        "--validation", choices=["full", "state", "none"], default="full"
    )
    p.add_argument(
        "--eager", action="store_true", help="omit whole-step MLX compilation"
    )
    p.add_argument("--cache-gib", type=float, default=16)
    p.add_argument(
        "--verify",
        choices=["cpu", "mps"],
        help="compare every loss, raw accumulated gradient and state to pinned upstream PyTorch outside timers",
    )
    p.add_argument(
        "--checkpoint",
        action="store_true",
        help="export final weights and optimizer state to NPZ outside timers",
    )
    p.add_argument("--eval-every", type=int, default=125)
    p.add_argument(
        "--val-tokens",
        type=int,
        help="default: 10*2**20 in quality mode, zero in throughput mode",
    )
    p.add_argument("--target-loss", type=float, default=3.28)
    p.add_argument("--write-sql", action="store_true")
    return p


def _download(directory, train_shards=1):
    directory.mkdir(parents=True, exist_ok=True)
    names = ["fineweb_val_000000.bin"] + [
        f"fineweb_train_{i:06d}.bin" for i in range(1, train_shards + 1)
    ]
    for name in names:
        path = directory / name
        if path.exists():
            continue
        temporary = path.with_suffix(".download")
        url = (
            "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/"
            + name
        )
        print(
            json.dumps({"kind": "download", "url": url, "path": str(path)}),
            flush=True,
        )
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)


def _hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _check(actual, expected, label, *, atol=3e-5, rtol=3e-5):
    if actual.keys() != expected.keys():
        raise AssertionError("oracle array names differ")
    maximum = 0.0
    for name, reference in expected.items():
        np.testing.assert_allclose(
            actual[name],
            reference,
            atol=atol,
            rtol=rtol,
            err_msg=f"{label}: {name}",
        )
        maximum = max(maximum, float(np.max(np.abs(actual[name] - reference))))
    return maximum


def main(argv=None):
    started = time.perf_counter()
    parser = _parser()
    args = parser.parse_args(argv)
    base = (
        {
            "layers": 1,
            "heads": 2,
            "width": 16,
            "vocab": 64,
            "context": 8,
            "batch": 2,
        }
        if args.preset == "tiny"
        else {}
    )
    for key in _Config.__dataclass_fields__:
        if getattr(args, key, None) is not None:
            base[key] = getattr(args, key)
    try:
        config = _Config(**base)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.warmup < args.steps <= config.iterations:
        parser.error("require 0 <= warmup < steps <= iterations")
    if (
        not math.isfinite(args.cache_gib)
        or args.cache_gib <= 0
        or args.eval_every < 1
    ):
        parser.error("positive cache size and evaluation interval required")
    if not math.isfinite(args.target_loss) or args.target_loss <= 0:
        parser.error("positive finite target loss required")
    if not 1 <= args.download_train_shards <= 103:
        parser.error("download-train-shards must be in [1, 103]")
    if args.synthetic and (
        args.data_dir or args.download_data or args.mode == "quality"
    ):
        parser.error(
            "synthetic data requires throughput mode without FineWeb inputs"
        )
    if not args.synthetic and args.data_dir is None:
        parser.error(
            "provide --data-dir for FineWeb, or explicitly choose --synthetic"
        )
    val_tokens = (
        args.val_tokens
        if args.val_tokens is not None
        else (10 * 2**20 if args.mode == "quality" else 0)
    )
    batch_tokens = config.batch * config.context
    if (
        val_tokens < 0
        or val_tokens % batch_tokens
        or (args.mode == "quality" and not val_tokens)
    ):
        parser.error(
            "validation token count must be a positive batch multiple in quality mode"
        )
    if args.synthetic and val_tokens:
        parser.error("synthetic throughput has no FineWeb validation loss")
    if args.backend == "sql_mlx" and os.environ.get("MLX_ENABLE_TF32") != "0":
        parser.error("SQL MLX requires MLX_ENABLE_TF32=0 before Python starts")
    if (
        args.verify or args.backend.startswith("torch")
    ) and _TorchTrainer is None:
        parser.error(
            "the PyTorch control/oracle requires torch in this interpreter"
        )
    if args.verify and not args.backend.startswith("sql"):
        parser.error(
            "--verify runs the SQL backend against the independent PyTorch control"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "metrics.jsonl").exists():
        parser.error(
            "use a fresh output directory; metrics.jsonl already exists"
        )
    if args.download_data:
        _download(args.data_dir, args.download_train_shards)
    streams, files = {}, []
    if not args.synthetic:
        for split in ("train", "val") if val_tokens else ("train",):
            paths = sorted(
                Path(p)
                for p in glob.glob(
                    str(args.data_dir / f"fineweb_{split}_*.bin")
                )
            )
            streams[split] = _ShardStream(paths, config)
            files.extend(paths)
        if val_tokens and val_tokens > sum(
            (len(a) - 1) // batch_tokens * batch_tokens
            for a in streams["val"].arrays
        ):
            parser.error(
                "validation request would wrap/repeat the FineWeb validation stream"
            )
        usable = sum(
            (len(a) - 1) // batch_tokens * batch_tokens
            for a in streams["train"].arrays
        )
        if (
            args.mode == "quality"
            and args.steps * config.accumulation * batch_tokens > usable
        ):
            parser.error(
                "quality run needs enough training shards to avoid repeating the data"
            )
    rng = np.random.default_rng(config.seed + 1)
    params = _initial(config)
    if args.backend.startswith("sql"):
        trainer = _SQLTrainer(
            config,
            params,
            gpu=args.backend == "sql_mlx",
            compiled=not args.eager,
            verify=bool(args.verify),
            validation=args.validation,
            cache_gib=args.cache_gib,
        )
    else:
        trainer = _TorchTrainer(
            config, params, args.upstream, args.backend.removeprefix("torch_")
        )
    oracle = (
        _TorchTrainer(config, params, args.upstream, args.verify)
        if args.verify
        else None
    )
    manifest = config.manifest() | {
        "precision": "float64-port"
        if args.backend == "sql_cpu"
        else "float32-port",
        "backend": args.backend,
        "data": "synthetic" if args.synthetic else "FineWeb10B-GPT2",
        "mode": args.mode,
        "steps": args.steps,
        "warmup_updates": args.warmup,
        "validation": args.validation
        if args.backend.startswith("sql")
        else "native",
        "compiled": args.backend == "sql_mlx" and not args.eager,
        "val_tokens": val_tokens,
        "target_loss": args.target_loss,
        "official_speedrun_comparable": False,
        "differences_from_record": [
            "float32 instead of CUDA BF16/TF32",
            "one device instead of eight H100s",
            "sequential batches without the original initial-batch replay",
        ],
        "data_shards": [
            {
                "path": str(p.resolve()),
                "bytes": p.stat().st_size,
                "sha256": _hash(p),
            }
            for p in files
        ],
        "initial_state_sha256": {
            n: hashlib.sha256(w.tobytes()).hexdigest()
            for n, w in params.items()
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": {
            "cpu": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if platform.system() == "Darwin"
            else platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "datafusion", "mlx", "torch")
            if importlib.util.find_spec(n) is not None
        },
        "implementation_sha256": {
            p.name: _hash(p) for p in Path(__file__).parent.glob("*modded*.py")
        },
    }
    if config.batch * config.accumulation != 512:
        manifest["differences_from_record"].append(
            "effective sequence batch differs from 512"
        )
    if config.context != 1024:
        manifest["differences_from_record"].append("context differs from 1024")
    if (config.layers, config.heads, config.width, config.vocab) != (
        12,
        6,
        768,
        50304,
    ):
        manifest["differences_from_record"].append(
            "model dimensions differ from historical 124M recipe"
        )
    if args.write_sql:
        (args.out_dir / "model.sql").write_text(_model(config))
        (args.out_dir / "gradients.sql").write_text(
            ";\n\n".join(_gradients(config).values())
        )
        if args.backend.startswith("sql"):
            for filename, queries in (
                ("stages.sql", trainer.final.stages),
                ("updates.sql", trainer.final.updates),
            ):
                (args.out_dir / filename).write_text(
                    ";\n\n".join(
                        f"-- {name}\n{query}" for name, query in queries.items()
                    )
                )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"kind": "setup", **manifest}), flush=True)
    setup_seconds = time.perf_counter() - started
    records, evaluations = [], []
    training_seconds, target_time = 0.0, None
    maximum_error = 0.0
    try:
        with (args.out_dir / "metrics.jsonl").open("w") as log:

            def emit(event):
                log.write(json.dumps(event) + "\n")
                log.flush()
                print(json.dumps(event), flush=True)

            def evaluate(iteration):
                nonlocal target_time
                streams["val"].reset()
                begin = time.perf_counter()
                losses = [
                    trainer.evaluate(*streams["val"].next())
                    for _ in range(val_tokens // batch_tokens)
                ]
                value = statistics.mean(losses)
                if not math.isfinite(value):
                    raise ValueError("nonfinite validation loss")
                if value <= args.target_loss and target_time is None:
                    target_time = training_seconds
                event = {
                    "kind": "eval",
                    "updates": iteration,
                    "loss": value,
                    "tokens": val_tokens,
                    "seconds": time.perf_counter() - begin,
                    "training_seconds": training_seconds,
                }
                evaluations.append(event)
                emit(event)

            if val_tokens:
                evaluate(0)
            for iteration in range(args.steps):
                begin = time.perf_counter()
                trainer.schedule(iteration)
                batches, losses, result, measurements = [], [], None, {}
                for micro in range(config.accumulation):
                    pair = (
                        tuple(
                            rng.integers(
                                min(config.vocab, 50257),
                                size=(config.batch, config.context),
                                dtype=np.int64,
                            )
                            for _ in range(2)
                        )
                        if args.synthetic
                        else streams["train"].next()
                    )
                    if oracle:
                        batches.append(pair)
                    result = trainer.microstep(
                        *pair, final=micro == config.accumulation - 1
                    )
                    losses.append(
                        result.loss_value()
                        if args.backend.startswith("sql")
                        else result
                    )
                    if args.backend.startswith("sql"):
                        for name, count in result.measurements.items():
                            measurements[name] = (
                                measurements.get(name, 0) + count
                            )
                seconds = time.perf_counter() - begin
                loss = statistics.mean(losses)
                if not math.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                training_seconds += seconds
                event = {
                    "kind": "train",
                    "iteration": iteration,
                    "loss": loss,
                    "lr": config.lr(iteration),
                    "seconds": seconds,
                    "training_seconds": training_seconds,
                    "tokens_per_second": config.manifest()["tokens_per_update"]
                    / seconds,
                }
                if measurements:
                    event["device_measurements"] = measurements
                if oracle:
                    oracle.schedule(iteration)
                    expected_losses = [
                        oracle.microstep(*pair, final=i == len(batches) - 1)
                        for i, pair in enumerate(batches)
                    ]
                    np.testing.assert_allclose(
                        losses,
                        expected_losses,
                        atol=3e-5,
                        rtol=3e-5,
                        err_msg="microbatch loss",
                    )
                    maximum_error = max(
                        maximum_error,
                        _check(
                            trainer.arrays(result, gradients=True),
                            oracle.arrays(gradients=True),
                            f"update {iteration}",
                        ),
                    )
                    event["oracle_maximum_absolute_error"] = maximum_error
                records.append(event)
                emit(event)
                if val_tokens and (
                    (iteration + 1) % args.eval_every == 0
                    or iteration + 1 == args.steps
                ):
                    evaluate(iteration + 1)
            training_statistics = (
                trainer.session.statistics()
                if args.backend.startswith("sql")
                else {}
            )
            if args.checkpoint:
                arrays = trainer.arrays()
                np.savez(
                    args.out_dir / "last.npz",
                    metadata=np.array(json.dumps(manifest)),
                    **arrays,
                )
            summary = manifest | {
                "setup_seconds": setup_seconds,
                "first_update_seconds": records[0]["seconds"],
                "training_seconds": training_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "warm_seconds_per_update": statistics.mean(
                    r["seconds"] for r in records[args.warmup :]
                ),
                "warm_tokens_per_second": config.manifest()["tokens_per_update"]
                / statistics.mean(r["seconds"] for r in records[args.warmup :]),
                "time_to_target_training_seconds": target_time,
                "oracle": args.verify,
                "oracle_maximum_absolute_error": maximum_error
                if oracle
                else None,
                "evaluations": evaluations,
                "statistics_before_checkpoint": training_statistics,
                "peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
            }
            (args.out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            emit(
                {
                    "kind": "complete",
                    "updates": len(records),
                    "warm_ms": summary["warm_seconds_per_update"] * 1000,
                    "tokens_per_second": summary["warm_tokens_per_second"],
                    "target_reached": target_time is not None,
                }
            )
    finally:
        trainer.close()
        if oracle:
            oracle.close()


if __name__ == "__main__":
    main()
