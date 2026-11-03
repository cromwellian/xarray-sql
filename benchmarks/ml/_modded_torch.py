"""Execute the pinned upstream model and optimizer as a float32 control.

The MIT-licensed Modded-NanoGPT source is read from a user-selected checkout.
Only definitions are loaded; its CUDA/distributed main program never runs.
The two bfloat16 casts in rotary/orthogonalization become float32, and
torch.compile is omitted for the local eager MPS control.
"""

from __future__ import annotations

import ast
import hashlib
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from _modded_nanogpt import _SOURCE, _SOURCE_SHA256
from torch import nn


class _Float32(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if node.attr == "bfloat16":
            node.attr = "float"
        return node


def _definitions(checkout):
    path = checkout / _SOURCE
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != _SOURCE_SHA256:
        raise ValueError(f"upstream source differs from pinned recipe: {path}")
    body = []
    # The pinned file's docstring has an invalid escape on newer Python.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Hyperparameters":
            break
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            body.append(_Float32().visit(node))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "zeropower_backends"
            for t in node.targets
        ):
            body.append(node)
    scope = {
        "__name__": __name__,
        "torch": torch,
        "nn": nn,
        "F": F,
        "dataclass": dataclass,
    }
    exec(  # noqa: S102 -- Hash-checked, explicitly selected upstream definitions.
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope


class _TorchTrainer:
    def __init__(self, config, params, checkout, device):
        self.config, self.device = config, torch.device(device)
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("PyTorch MPS is unavailable")
        definitions = _definitions(checkout)
        model_config = definitions["GPTConfig"](
            vocab_size=config.vocab,
            n_layer=config.layers,
            n_head=config.heads,
            n_embd=config.width,
        )
        self.model = definitions["GPT"](model_config).to(self.device)
        self.parameters = {"wte": self.model.lm_head.weight}
        for i, block in enumerate(self.model.transformer.h):
            for suffix, layer in (
                ("wq", block.attn.c_q),
                ("wk", block.attn.c_k),
                ("wv", block.attn.c_v),
                ("wo", block.attn.c_proj),
                ("fc", block.mlp.c_fc),
                ("proj", block.mlp.c_proj),
            ):
                self.parameters[f"l{i}_{suffix}"] = layer.weight
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(torch.tensor(params[name], device=self.device))
        self.adam = torch.optim.AdamW(
            self.model.lm_head.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        )
        self.muon = definitions["Muon"](
            self.model.transformer.h.parameters(),
            lr=config.learning_rate * 0.1,
            momentum=0.95,
            backend_steps=config.ns_steps,
        )
        self._last_grads = {}
        self.synchronize()

    def synchronize(self):
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def schedule(self, iteration):
        self.adam.param_groups[0]["lr"] = self.config.lr(iteration)
        self.muon.param_groups[0]["lr"] = self.config.lr(iteration) * 0.1

    def microstep(self, tok, nxt, final):
        x, y = (torch.tensor(a, device=self.device) for a in (tok, nxt))
        _, loss = self.model(x, y, return_logits=False)
        (loss / self.config.accumulation).backward()
        if final:
            # Keep references for the optional oracle; zero_grad releases them
            # after stepping, rather than modifying their storage.
            self._last_grads = {
                name: p.grad for name, p in self.parameters.items()
            }
            self.adam.step()
            self.muon.step()
            self.model.zero_grad(set_to_none=True)
        self.synchronize()
        return float(loss.detach())

    def evaluate(self, tok, nxt):
        with torch.no_grad():
            _, loss = self.model(
                *(torch.tensor(a, device=self.device) for a in (tok, nxt)),
                return_logits=False,
            )
        self.synchronize()
        return float(loss)

    def arrays(self, gradients=False):
        result = {}
        for name, parameter in self.parameters.items():
            result[name + "_w"] = parameter.detach().cpu().numpy().copy()
            state = (
                self.adam.state[parameter]
                if name == "wte"
                else self.muon.state[parameter]
            )
            for output, source in (
                (("m", "exp_avg"), ("v", "exp_avg_sq"))
                if name == "wte"
                else (("m", "momentum_buffer"),)
            ):
                result[name + "_" + output] = (
                    state[source].detach().cpu().numpy().copy()
                )
            if gradients:
                result[name + "_g"] = (
                    self._last_grads[name].detach().cpu().numpy().copy()
                )
        return result

    def close(self):
        pass
