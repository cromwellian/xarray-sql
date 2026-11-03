"""SQL form of a pinned Modded-NanoGPT training algorithm.

Modded-NanoGPT (https://github.com/KellerJordan/modded-nanogpt) is the
FineWeb language-model training speedrun. This implements its October 14,
2024 architecture and optimizer equations in float32, without CUDA-specific
mixed precision. ddx (https://github.com/xqlsystems/ddx) differentiates the
loss; the optimizer is ordinary SQL, not differentiated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr

_REVISION = "4ea6b937337a4889b8cfe3f38a93d120048d8f71"
_SOURCE = "records/track_1_short/2024-10-14_ModernArch/train_gpt2.py"
_SOURCE_SHA256 = (
    "7199ba9f956c20aceffc7112a0e430db68964fa19d4fa427f743fc3d5f33bd4b"
)
_EPS = float(np.finfo(np.float32).eps)


@dataclass(frozen=True)
class _Config:
    layers: int = 12
    heads: int = 6
    width: int = 768
    vocab: int = 50304
    context: int = 1024
    batch: int = 1
    accumulation: int = 1
    iterations: int = 5100
    warmdown: int = 1450
    learning_rate: float = 0.0036
    ns_steps: int = 5
    seed: int = 1337

    def __post_init__(self):
        integers = (
            self.layers,
            self.heads,
            self.width,
            self.vocab,
            self.context,
            self.batch,
            self.accumulation,
            self.iterations,
            self.ns_steps,
        )
        if min(integers) < 1 or self.width % (2 * self.heads):
            raise ValueError(
                "positive sizes and an even integral head width are required"
            )
        if not 0 < self.warmdown <= self.iterations:
            raise ValueError("warmdown must be in [1, iterations]")
        if not np.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning rate must be finite and nonnegative")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32 value")

    @property
    def head(self):
        return self.width // self.heads

    @property
    def shapes(self):
        result = {"wte": (self.vocab, self.width)}
        for layer in range(self.layers):
            for suffix in ("wq", "wk", "wv", "wo", "fc", "proj"):
                result[f"l{layer}_{suffix}"] = (
                    (4 * self.width, self.width)
                    if suffix == "fc"
                    else (self.width, 4 * self.width)
                    if suffix == "proj"
                    else (self.width, self.width)
                )
        return result

    def lr(self, iteration):
        if not 0 <= iteration < self.iterations:
            raise ValueError(
                "iteration outside the original learning-rate schedule"
            )
        return self.learning_rate * min(
            1.0, (self.iterations - iteration) / self.warmdown
        )

    def manifest(self):
        return asdict(self) | {
            "parameters": sum(int(np.prod(s)) for s in self.shapes.values()),
            "tokens_per_update": self.batch * self.context * self.accumulation,
            "precision": "float32-port",
            "upstream_revision": _REVISION,
            "upstream_source": _SOURCE,
            "upstream_source_sha256": _SOURCE_SHA256,
        }


def _initial(config):
    rng = np.random.default_rng(config.seed)
    params = {}
    for name, shape in config.shapes.items():
        limit = shape[1] ** -0.5
        params[name] = (
            np.zeros(shape, dtype=np.float32)
            if name.endswith(("_wo", "_proj"))
            else rng.uniform(-limit, limit, shape).astype(np.float32)
        )
    return params


def _table(name, **columns):
    dims = ("tok", "e") if name == "wte" else ("i", "j")
    shape = next(iter(columns.values())).shape
    return xr.Dataset(
        {
            key: (dims, value.astype(np.float64))
            for key, value in columns.items()
        },
        coords={dim: np.arange(size) for dim, size in zip(dims, shape)},
    )


def _batch(tok, nxt):
    return xr.Dataset(
        {"tok": (("seq", "pos"), tok), "next": (("seq", "pos"), nxt)},
        coords={"seq": np.arange(len(tok)), "pos": np.arange(tok.shape[1])},
    )


def _norm(source, target, keys, feature):
    group = ",".join(keys)
    same = " AND ".join(f"x.{k}=n.{k}" for k in keys)
    fields = ",".join(f"x.{k}" for k in [*keys, feature])
    return f"""{target}_norm AS (
      SELECT {group}, sqrt(AVG(val*val)+{_EPS}) AS n FROM {source}
      GROUP BY {group}),
    {target} AS (
      SELECT {fields}, x.val/n.n AS val FROM {source} x
      JOIN {target}_norm n ON {same}),"""


def _linear(source, target, weights, feature="e", output="e"):
    return f"""{target} AS (
      SELECT x.seq,x.pos,w.i AS {output},SUM(x.val*w.w) AS val
      FROM {source} x JOIN {weights} w ON x.{feature}=w.j
      GROUP BY x.seq,x.pos,w.i),"""


def _residual(left, right, target):
    return f"""{target} AS (
      SELECT x.seq,x.pos,x.e,x.val+y.val AS val FROM {left} x JOIN {right} y
      ON x.seq=y.seq AND x.pos=y.pos AND x.e=y.e),"""


def _model(config):
    chunks = [
        "x0 AS (SELECT b.seq,b.pos,w.e,w.w AS val FROM batch b JOIN wte w ON b.tok=w.tok),"
    ]
    for layer in range(config.layers):
        l = f"l{layer}"
        chunks.append(_norm(f"x{layer}", f"{l}_n1", ["seq", "pos"], "e"))
        for kind in ("q", "k", "v"):
            raw = f"{l}_{kind}_raw"
            chunks.append(_linear(f"{l}_n1", raw, f"{l}_w{kind}"))
            if kind in ("q", "k"):
                chunks.append(f"""{l}_{kind}_rot AS (
                  SELECT x.seq,x.pos,r.h,r.c,x.val*r.co+y.val*r.si AS val
                  FROM {raw} x JOIN rotary r ON x.pos=r.pos AND x.e=r.e
                  JOIN {raw} y ON x.seq=y.seq AND x.pos=y.pos AND y.e=r.partner),""")
                chunks.append(
                    _norm(
                        f"{l}_{kind}_rot",
                        f"{l}_{kind}",
                        ["seq", "pos", "h"],
                        "c",
                    )
                )
            else:
                chunks.append(f"""{l}_v AS (
                  SELECT x.seq,x.pos,r.h,r.c,x.val FROM {raw} x
                  JOIN head_map r ON x.e=r.e),""")
        chunks.append(f"""{l}_scores AS (
          SELECT q.seq,q.h,q.pos AS i,k.pos AS j,SUM(q.val*k.val)/{config.head**0.5} AS s
          FROM {l}_q q JOIN {l}_k k ON q.seq=k.seq AND q.h=k.h AND q.c=k.c
          WHERE q.pos>=k.pos GROUP BY q.seq,q.h,q.pos,k.pos),
        {l}_maximum AS (
          SELECT seq,h,i,ddx_stop_gradient(MAX(s)) AS m FROM {l}_scores GROUP BY seq,h,i),
        {l}_weights AS (
          SELECT s.seq,s.h,s.i,s.j,exp(s.s-m.m) AS p FROM {l}_scores s
          JOIN {l}_maximum m ON s.seq=m.seq AND s.h=m.h AND s.i=m.i),
        {l}_denom AS (SELECT seq,h,i,SUM(p) AS z FROM {l}_weights GROUP BY seq,h,i),
        {l}_heads AS (
          SELECT a.seq,a.i AS pos,a.h,v.c,SUM(a.p/d.z*v.val) AS val
          FROM {l}_weights a JOIN {l}_denom d ON a.seq=d.seq AND a.h=d.h AND a.i=d.i
          JOIN {l}_v v ON a.seq=v.seq AND a.h=v.h AND a.j=v.pos
          GROUP BY a.seq,a.i,a.h,v.c),
        {l}_joined AS (
          SELECT x.seq,x.pos,r.e,x.val FROM {l}_heads x
          JOIN head_map r ON x.h=r.h AND x.c=r.c),""")
        chunks.append(_linear(f"{l}_joined", f"{l}_attn", f"{l}_wo"))
        chunks.append(_residual(f"x{layer}", f"{l}_attn", f"{l}_mid"))
        chunks.append(_norm(f"{l}_mid", f"{l}_n2", ["seq", "pos"], "e"))
        chunks.append(_linear(f"{l}_n2", f"{l}_fc_raw", f"{l}_fc", output="f"))
        chunks.append(f"""{l}_relu AS (
          SELECT seq,pos,f,CASE WHEN val>0 THEN val*val ELSE 0.0 END AS val
          FROM {l}_fc_raw),""")
        chunks.append(
            _linear(f"{l}_relu", f"{l}_mlp", f"{l}_proj", feature="f")
        )
        chunks.append(_residual(f"{l}_mid", f"{l}_mlp", f"x{layer + 1}"))
    chunks.append(
        _norm(f"x{config.layers}", "final", ["seq", "pos"], "e").rstrip(",")
    )
    return (
        "WITH "
        + "\n".join(chunks)
        + """
    SELECT x.seq,x.pos,w.tok,SUM(x.val*w.w) AS val FROM final x
    JOIN wte w ON x.e=w.e GROUP BY x.seq,x.pos,w.tok"""
    )


_LOSS = """predicted AS (SELECT seq,pos,tok,val FROM logits),
maximum AS (SELECT seq,pos,ddx_stop_gradient(MAX(val)) AS m FROM predicted GROUP BY seq,pos),
lse AS (SELECT p.seq,p.pos,MAX(m.m)+ln(SUM(exp(p.val-m.m))) AS z
  FROM predicted p JOIN maximum m ON p.seq=m.seq AND p.pos=m.pos GROUP BY p.seq,p.pos),
loss AS (SELECT AVG(s.z-p.val) AS l FROM lse s JOIN batch b ON s.seq=b.seq AND s.pos=b.pos
  JOIN predicted p ON p.seq=b.seq AND p.pos=b.pos AND p.tok=b.next)"""


def _gradients(config):
    return {
        name
        + "_grad": f"WITH {_LOSS} SELECT {','.join(_dims(name))},grad(loss,w) AS g FROM {name} ORDER BY {','.join(_dims(name))}"
        for name in config.shapes
    }


def _dims(name):
    return ("tok", "e") if name == "wte" else ("i", "j")


def _muon(name, shape, steps, gradient):
    transpose = shape[0] > shape[1]
    xi, xj = ("j", "i") if transpose else ("i", "j")
    moment = name + "_moment"
    stages = {
        moment: f"""SELECT p.i,p.j,0.95*s.m+g.g AS m,g.g FROM {name} p
      JOIN {name}_state s ON p.i=s.i AND p.j=s.j
      JOIN {gradient} g ON p.i=g.i AND p.j=g.j ORDER BY p.i,p.j"""
    }
    stages[name + "_ns0"] = f"""WITH nesterov AS (
      SELECT i,j,g+0.95*m AS u FROM {moment}),
      magnitude AS (SELECT sqrt(SUM(u*u))+1e-7 AS n FROM nesterov)
      SELECT u.{xi} AS i,u.{xj} AS j,u.u/n.n AS x FROM nesterov u CROSS JOIN magnitude n
      ORDER BY i,j"""
    # Explicit device stages bound SQL expansion at every polynomial iteration.
    # They stay in the same compiled GPU training step and do not export Arrow.
    for k in range(steps):
        source = name + f"_ns{k}"
        stages[name + f"_ns{k + 1}"] = f"""WITH a AS (
          SELECT x.i,y.i AS j,SUM(x.x*y.x) AS x FROM {source} x JOIN {source} y ON x.j=y.j GROUP BY x.i,y.i),
        b AS (
          SELECT a.i,x.j,SUM(a.x*x.x) AS x FROM a JOIN {source} x ON a.j=x.i GROUP BY a.i,x.j),
        c AS (
          SELECT a.i,b.j,SUM(a.x*b.x) AS x FROM a JOIN b ON a.j=b.i GROUP BY a.i,b.j)
        SELECT x.i,x.j,3.4445*x.x-4.775*b.x+2.0315*c.x AS x FROM {source} x
          JOIN b ON x.i=b.i AND x.j=b.j JOIN c ON x.i=c.i AND x.j=c.j ORDER BY x.i,x.j"""
    update = f"""SELECT p.i,p.j,p.w-t.lr*0.1*{max(shape) ** 0.5}*x.x AS w,m.m
      FROM {name} p JOIN {moment} m ON p.i=m.i AND p.j=m.j
      JOIN {name}_ns{steps} x ON p.i=x.{xi} AND p.j=x.{xj}
      CROSS JOIN train_step t ORDER BY p.i,p.j"""
    return stages, update


def _adam(gradient):
    return f"""WITH moments AS (
      SELECT p.tok,p.e,p.w,0.9*s.m+0.1*g.g AS m,0.95*s.v+0.05*g.g*g.g AS v
      FROM wte p JOIN wte_state s ON p.tok=s.tok AND p.e=s.e
      JOIN {gradient} g ON p.tok=g.tok AND p.e=g.e)
    SELECT p.tok,p.e,p.w-t.lr*(p.m/t.b1)/(sqrt(p.v/t.b2)+1e-8) AS w,p.m,p.v
    FROM moments p CROSS JOIN train_step t ORDER BY p.tok,p.e"""


def _static(config):
    h, c = np.arange(config.heads), np.arange(config.head)
    e = h[:, None] * config.head + c[None, :]
    half = config.head // 2
    frequency = 1.0 / np.power(
        np.float32(10000),
        np.arange(0, config.head, 2, dtype=np.float32) / config.head,
    )
    angles = (
        np.arange(config.context, dtype=np.float32)[:, None]
        * frequency[None, :]
    )
    co = np.cos(angles)[:, c % half]
    si = np.sin(angles)[:, c % half] * np.where(c < half, 1, -1)[None, :]
    return {
        "head_map": xr.Dataset({"e": (("h", "c"), e)}, coords={"h": h, "c": c}),
        "rotary": xr.Dataset(
            {
                "e": (
                    ("pos", "h", "c"),
                    np.broadcast_to(
                        e, (config.context, config.heads, config.head)
                    ),
                ),
                "partner": (
                    ("pos", "h", "c"),
                    np.broadcast_to(
                        h[:, None] * config.head
                        + (c[None, :] + half) % config.head,
                        (config.context, config.heads, config.head),
                    ),
                ),
                "co": (
                    ("pos", "h", "c"),
                    np.broadcast_to(
                        co[:, None, :],
                        (config.context, config.heads, config.head),
                    ).astype(np.float64),
                ),
                "si": (
                    ("pos", "h", "c"),
                    np.broadcast_to(
                        si[:, None, :],
                        (config.context, config.heads, config.head),
                    ).astype(np.float64),
                ),
            },
            coords={"pos": np.arange(config.context), "h": h, "c": c},
        ),
    }


class _ShardStream:
    """Sequential uint16 FineWeb shards with the upstream llm.c header."""

    def __init__(self, files, config):
        self.files, self.config = sorted(map(Path, files)), config
        if not self.files:
            raise ValueError("no FineWeb shards matched")
        self.arrays = []
        for path in self.files:
            header = np.fromfile(path, dtype="<i4", count=256)
            if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
                raise ValueError(f"invalid FineWeb header: {path}")
            if (
                header[2] <= config.batch * config.context
                or path.stat().st_size != 1024 + int(header[2]) * 2
            ):
                raise ValueError(f"invalid or too short FineWeb shard: {path}")
            self.arrays.append(
                np.memmap(
                    path,
                    dtype="<u2",
                    mode="r",
                    offset=1024,
                    shape=(int(header[2]),),
                )
            )
        self.reset()

    def reset(self):
        self.shard = self.position = 0

    def next(self):
        count = self.config.batch * self.config.context
        tokens = self.arrays[self.shard][
            self.position : self.position + count + 1
        ].astype(np.int64)
        if np.any(tokens >= min(self.config.vocab, 50257)):
            raise ValueError("token outside GPT-2/model vocabulary")
        self.position += count
        if self.position + count + 1 > len(self.arrays[self.shard]):
            self.shard = (self.shard + 1) % len(self.arrays)
            self.position = 0
        shape = (self.config.batch, self.config.context)
        return tokens[:-1].reshape(shape), tokens[1:].reshape(shape)
