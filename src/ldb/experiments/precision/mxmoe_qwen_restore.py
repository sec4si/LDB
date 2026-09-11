"""Restore packed Qwen weights without loading or quantizing the FP16 source."""

import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .helpers import sha
from .mxmoe_bridge import Bridge
from .mxmoe_model import PaperBank


def restore(checkpoint, library, device="cuda"):
    freeze = json.loads((checkpoint / "freeze.json").read_text())
    if sha(library) != freeze["library_sha256"]:
        raise ValueError("Checkpoint kernel library mismatch")
    allocation = json.loads((checkpoint / "allocation.json").read_text())
    plan = allocation["plan"]
    kind = allocation.get("format_kind", "weight_only")
    if kind not in ("weight_only", "activation_mixed", "linear_mixed"):
        raise ValueError("Unknown format kind")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    state = torch.load(checkpoint / "state.pt", map_location="cpu", mmap=True, weights_only=True)
    bridge = Bridge(library)
    kernel = next(
        i for i, n in enumerate(bridge.names) if n.endswith("groupgemm_hz_fused_0_cached")
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.float16, attn_implementation="eager", experts_implementation="eager"
        )
        layers = [
            layer_row for layer_row in model.model.layers if hasattr(layer_row.mlp, "experts")
        ]
        if len(layers) != len(plan):
            raise ValueError("Incomplete layer plan")
        for layer, choices in zip(layers, plan):
            li = list(model.model.layers).index(layer)
            rotation = state.get(f"model.layers.{li}.mlp.experts.down_rotation")
            if rotation is not None:
                rotation = torch.empty_like(rotation, device="meta")
            if kind == "linear_mixed":
                from .mxmoe_linear_bank import LinearBank

                if rotation is not None:
                    raise ValueError("Online rotation unsupported for mixed activation bank")
                layer.mlp.experts = LinearBank(layer.mlp.experts, choices, bridge, kernel)
            elif kind == "activation_mixed":
                from .mxmoe_activation_bank import ActivationBank

                if rotation is not None:
                    raise ValueError("Online rotation unsupported for mixed activation bank")
                layer.mlp.experts = ActivationBank(layer.mlp.experts, choices, bridge, kernel)
            else:
                layer.mlp.experts = PaperBank(layer.mlp.experts, choices, bridge, kernel, rotation)
    model.load_state_dict(state, strict=True, assign=True)
    # Nonpersistent rotary buffers are absent from state_dict; initialize on CPU.
    for module in model.modules():
        if hasattr(module, "inv_freq") and module.inv_freq.is_meta:
            replacement = type(module)(config, device="cpu")
            module.register_buffer("inv_freq", replacement.inv_freq, persistent=False)
            module.original_inv_freq = replacement.original_inv_freq
    return model.eval().to(device)
