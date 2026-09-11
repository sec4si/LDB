"""Frozen SwiGLU bodies and reversible intervention at the routed expert bank."""

from contextlib import contextmanager

import torch
from safetensors import safe_open


def body(x, weights):
    gate, up = torch.nn.functional.linear(x, weights[0]).chunk(2, dim=-1)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, weights[1])


def weights(record, layer, expert):
    with safe_open(record["path"] + "/model.safetensors", framework="pt") as stream:
        prefix = f"model.layers.{layer}.block_sparse_moe.experts.{expert}"
        gate, up, down = [
            stream.get_tensor(f"{prefix}.{n}.weight").float() for n in ["w1", "w3", "w2"]
        ]
    return torch.cat([gate, up]), down


@contextmanager
def replace(model, replacements):
    """Zero native down matrices, then add routed replacement outputs once."""
    saved, handles = [], []
    grouped = {}
    try:
        for layer, expert, function in replacements:
            if (layer, expert) in {(a, b) for a, b, _ in saved}:
                raise ValueError("Duplicate replacement slot")
            bank = model.model.layers[layer].mlp.experts
            saved.append((layer, expert, bank.down_proj[expert].detach().clone()))
            with torch.no_grad():
                bank.down_proj[expert].zero_()
            grouped.setdefault(layer, []).append((expert, function))
        for layer, entries in grouped.items():

            def inject(module, args, output, entries=entries):
                x, ids, routing = args
                for expert, function in entries:
                    coefficient = ((ids == expert) * routing).sum(1)
                    active = coefficient != 0
                    if function is not None and active.any():
                        output = output.clone()
                        output[active] += function(x[active]) * coefficient[active, None]
                return output

            handles.append(model.model.layers[layer].mlp.experts.register_forward_hook(inject))
        yield
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for layer, expert, original in saved:
                bank = model.model.layers[layer].mlp.experts
                bank.down_proj[expert].copy_(original)
                if not torch.equal(bank.down_proj[expert], original):
                    raise RuntimeError("Expert restoration failed")


@contextmanager
def zero_layer(model, layer):
    handle = model.model.layers[layer].mlp.register_forward_hook(
        lambda m, a, out: torch.zeros_like(out)
    )
    try:
        yield
    finally:
        handle.remove()
