"""Packed groupwise expert quantization; eager reference execution, not a fast kernel."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from .helpers import PoolBank, StoredExpert, sha, write


class PackedMatrix(nn.Module):
    def __init__(self, weight, scale, shape, precision):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.shape = tuple(shape)
        self.precision = precision
        self.backend = "reference"

    @classmethod
    def encode(cls, weight, precision):
        if precision not in ("int4", "int8", "bf16"):
            raise ValueError(precision)
        shape = weight.shape
        if len(shape) != 2 or shape[1] % 64:
            raise ValueError("Expected a matrix with input width divisible by 64")
        if precision == "bf16":
            return cls(weight.to(torch.bfloat16).clone(), weight.new_empty(0), shape, precision)
        groups = weight.float().reshape(shape[0], -1, 64)
        limit = 7 if precision == "int4" else 127
        scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-20) / limit
        quant = (groups / scale).round().clamp(-limit, limit).to(torch.int8).flatten()
        if precision == "int4":
            unsigned = (quant.to(torch.int16) + 8).to(torch.uint8)
            quant = unsigned[::2] | (unsigned[1::2] << 4)
        return cls(quant.contiguous(), scale.contiguous(), shape, precision)

    def dense(self):
        if self.precision == "bf16":
            return self.weight.float()
        weight = self.weight
        if self.precision == "int4":
            weight = torch.stack((weight & 15, weight >> 4), dim=-1).flatten().float() - 8
        return (weight.float().reshape(self.shape[0], -1, 64) * self.scale).reshape(self.shape)

    def forward(self, x):
        if self.backend == "triton":
            from .precision_kernel import linear

            return linear(x, self)
        return nn.functional.linear(x, self.dense().to(x.dtype))


def convert(model, plan):
    if len(plan) != len(model.model.layers):
        raise ValueError("Plan must cover every layer")
    for layer, precisions in zip(model.model.layers, plan):
        old = layer.mlp.experts
        if len(precisions) != old.gate_up_proj.shape[0]:
            raise ValueError("Plan must cover every expert")
        layer.mlp.experts = PoolBank(
            [
                StoredExpert(
                    PackedMatrix.encode(old.gate_up_proj[e].detach(), p),
                    PackedMatrix.encode(old.down_proj[e].detach(), p),
                )
                for e, p in enumerate(precisions)
            ]
        )
    return model


def save(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    descriptors = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedMatrix):
            descriptors[name] = dict(shape=module.shape, precision=module.precision)
    save_file(
        {k: v.detach().contiguous() for k, v in model.state_dict().items()},
        str(directory / "weights.safetensors"),
    )
    write(
        directory / "manifest.json",
        dict(
            config=model.config.to_dict(),
            matrices=descriptors,
            weights_sha256=sha(directory / "weights.safetensors"),
        ),
    )


def restore(directory):
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text())
    assert sha(directory / "weights.safetensors") == meta["weights_sha256"]
    values = load_file(str(directory / "weights.safetensors"))
    config = dict(meta["config"])
    model_type = config.pop("model_type")
    config = AutoConfig.for_model(model_type, **config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation="eager", experts_implementation="eager"
        )
    for layer_row, layer in enumerate(model.model.layers):
        experts = []
        for e in range(config.num_local_experts):
            matrices = []
            for part in ("gu", "down"):
                name = f"model.layers.{layer_row}.mlp.experts.experts.{e}.{part}"
                matrices.append(
                    PackedMatrix(
                        values[name + ".weight"], values[name + ".scale"], **meta["matrices"][name]
                    )
                )
            experts.append(StoredExpert(*matrices))
        layer.mlp.experts = PoolBank(experts)
    model.load_state_dict(values, assign=True, strict=True)
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=config, device="cpu")
    return model.eval()
