"""Record actual layer contracts, paired states and per-document behavior."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from .common import batches, freeze, load, setup, write
from .model import weights


def characterize(x, routing, output, residual):
    x, routing, output, residual = [v.double() for v in (x, routing, output, residual)]
    usage = routing.sum(0)
    usage = usage / usage.sum().clamp_min(1e-30)
    probabilities = routing.sort(dim=1, descending=True).values
    centered = x - x.mean(0)
    gram = centered @ centered.T
    rank = float(gram.trace().square() / gram.square().sum().clamp_min(1e-30))
    return dict(
        input_rms=float(x.square().mean().sqrt()),
        output_rms=float(output.square().mean().sqrt()),
        residual_rms=float(residual.square().mean().sqrt()),
        update_ratio=float(
            output.square().mean().sqrt() / residual.square().mean().sqrt().clamp_min(1e-30)
        ),
        routing_entropy=float(-(usage * usage.clamp_min(1e-30).log()).sum() / np.log(len(usage))),
        routed_weight_gap=float((probabilities[:, 0] - probabilities[:, 1]).mean()),
        sampled_input_participation_rank=rank,
        coverage=(routing > 0).double().mean(0).tolist(),
        routing_mass=usage.tolist(),
    )


def capture(model, bb):
    n = len(model.model.layers)
    arrays = [dict(x=[], route=[], y=[], residual=[]) for _ in range(n)]
    rows = [[] for _ in range(n)]
    document_ids = []
    pending, native_residual = {}, {}
    handles = []
    for layer, module in enumerate(model.model.layers):

        def save_residual(m, args, kwargs, layer=layer):
            value = args[0] if args else kwargs["hidden_states"]
            native_residual[layer] = value.detach().reshape(-1, value.shape[-1])

        def save(m, args, output, layer=layer):
            x, ids, w = args
            route = torch.zeros((len(x), m.num_experts), dtype=x.dtype)
            route.scatter_add_(1, ids, w)
            pending[layer] = (x.detach(), route, output.detach())

        handles.append(module.register_forward_pre_hook(save_residual, with_kwargs=True))
        handles.append(module.mlp.experts.register_forward_hook(save))
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(bb):
                model(**batch, use_cache=False, output_router_logits=False)
                width = batch["input_ids"].shape[1]
                for j, mask in enumerate(batch["attention_mask"]):
                    positions = mask.nonzero().flatten()
                    chosen = positions[
                        torch.linspace(0, len(positions) - 1, min(16, len(positions)))
                        .round()
                        .long()
                    ]
                    idx = chosen + j * width
                    doc = 2 * batch_index + j
                    document_ids.extend([doc] * len(idx))
                    for layer in range(n):
                        values = [v[idx].clone() for v in (*pending[layer], native_residual[layer])]
                        for key, value in zip(["x", "route", "y", "residual"], values, strict=True):
                            arrays[layer][key].append(value)
                        rows[layer].append(
                            dict(document=doc, samples=len(idx), **characterize(*values))
                        )
    finally:
        for handle in handles:
            handle.remove()
    return [{k: torch.cat(v).numpy() for k, v in a.items()} for a in arrays], rows, document_ids


def run(root):
    records = setup(root)
    contracts = []
    for m, record in enumerate(records):
        start = time.monotonic()
        model = load(record)
        local = []
        for layer, module in enumerate(model.model.layers):
            bank = module.mlp.experts
            d, h = model.config.hidden_size, model.config.intermediate_size
            if tuple(bank.gate_up_proj.shape) != (bank.num_experts, 2 * h, d):
                raise ValueError("Unsupported packed input shape")
            if tuple(bank.down_proj.shape) != (bank.num_experts, d, h):
                raise ValueError("Unsupported output shape")
            if model.config.hidden_act != "silu":
                raise ValueError("Unsupported expert activation")
            for expert in range(bank.num_experts):
                gu, down = weights(record, layer, expert)
                if not (
                    torch.equal(gu, bank.gate_up_proj[expert])
                    and torch.equal(down, bank.down_proj[expert])
                ):
                    raise ValueError("Packed runtime weights differ from native checkpoint")
            local.append(
                dict(
                    model=m,
                    layer=layer,
                    relative_depth=layer / max(1, len(model.model.layers) - 1),
                    operator="down(silu(gate(x))*up(x))",
                    hidden_size=d,
                    intermediate_size=h,
                    experts=bank.num_experts,
                    top_k=model.config.num_experts_per_tok,
                    gate_up_shape=list(bank.gate_up_proj.shape),
                    down_shape=list(bank.down_proj.shape),
                    normalization="pre-attention and pre-MoE RMSNorm",
                    residual="additive",
                    shared_experts=0,
                    gate_bias=False,
                    up_bias=False,
                    down_bias=False,
                    runtime_class=type(module.mlp).__name__,
                    weights_verified=True,
                    body_parameters=3 * d * h,
                )
            )
        for split in ["fit", "selection", "heldout"]:
            directory = root / "captures" / f"model-{m}" / split
            directory.mkdir(parents=True, exist_ok=False)
            arrays, rows, document_ids = capture(model, batches(root, m, split))
            for layer, data in enumerate(arrays):
                np.savez(
                    directory / f"layer-{layer}.npz", **data, document=np.asarray(document_ids)
                )
            write(directory / "layer-documents.json", rows)
            print("Captured", m, split, len(document_ids), "positions per layer", flush=True)
        contracts.extend(local)
        del model, arrays
        gc.collect()
        print("Model complete", m, round(time.monotonic() - start, 2), "seconds", flush=True)
    write(root / "layer-contracts.json", contracts)
    freeze(
        root,
        ["layer-contracts.json"]
        + [str(p.relative_to(root)) for p in (root / "captures").rglob("*") if p.is_file()],
        "capture-freeze.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    run(parser.parse_args().root)
