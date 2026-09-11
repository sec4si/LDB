"""Explicit packed-Qwen CUDA adapter; this is not a general MoE IR backend."""

import ctypes
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from ldb.experiments.precision.helpers import sha
from ldb.experiments.precision.mxmoe_activation_bank import ActivationBank
from ldb.experiments.precision.mxmoe_bridge import Bridge


class SharedBank(ActivationBank):
    def __init__(self, experts, plan, bridge, kernel, mapping):
        super().__init__(experts, plan, bridge, kernel)
        if len(mapping) != len(plan) or any(
            type(e) is not int or not 0 <= e < len(plan) for e in mapping
        ):
            raise ValueError("Invalid router-slot mapping")
        if any(mapping[d] != d or plan[e] != plan[d] for e, d in enumerate(mapping)):
            raise ValueError("Require retained donors with identical quantization formats")
        self.mapping = mapping
        for e, donor in enumerate(mapping):
            if e != donor:
                for part in ("gu", "down"):
                    for suffix in ("packed", "scale"):
                        delattr(self, f"{part}_{e}_{suffix}")
        self.fused_activation_pack = True
        self.coalesce_metadata = True

    def matrix(self, part, expert):
        return super().matrix(part, self.mapping[expert])


def restore(checkpoint, root):
    checkpoint, root = Path(checkpoint), Path(root)
    ctypes.CDLL(str(root / "libfmt.so.11"), mode=ctypes.RTLD_GLOBAL)
    library = root / "libgen_lib.so"
    freeze = json.loads((checkpoint / "freeze.json").read_text())
    if sha(library) != freeze["library_sha256"]:
        raise ValueError("Kernel identity mismatch")
    allocation = json.loads((checkpoint / "allocation.json").read_text())
    plan = allocation["plan"]
    if allocation.get("format_kind", "weight_only") == "weight_only":
        plan = [
            [
                {
                    p: "w4a16" if (e[p] if isinstance(e, dict) else e) == 4 else "unsupported"
                    for p in ("gu", "down")
                }
                for e in layer
            ]
            for layer in plan
        ]
    mappings = allocation.get("sharing", [list(range(len(p))) for p in plan])
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    state_path = checkpoint / "state.pt"
    overlay = checkpoint / "state-reference.json"
    if overlay.exists():
        if state_path.exists():
            raise ValueError("Ambiguous overlay and physical checkpoint")
        reference = json.loads(overlay.read_text())
        state_path = Path(reference["path"])
        if sha(state_path) != reference["sha256"]:
            raise ValueError("Overlay source identity mismatch")
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    if overlay.exists():
        for li, mapping in enumerate(mappings):
            for expert, donor in enumerate(mapping):
                if expert != donor:
                    for part in ("gu", "down"):
                        for suffix in ("packed", "scale"):
                            del state[f"model.layers.{li}.mlp.experts.{part}_{expert}_{suffix}"]
    bridge = Bridge(library)
    kernel = next(
        i for i, name in enumerate(bridge.names) if name.endswith("groupgemm_hz_fused_0_cached")
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.float16, attn_implementation="eager", experts_implementation="eager"
        )
        if len(model.model.layers) != len(plan) or len(mappings) != len(plan):
            raise ValueError("Incomplete layer mapping")
        for layer, choices, mapping in zip(model.model.layers, plan, mappings, strict=True):
            layer.mlp.experts = SharedBank(layer.mlp.experts, choices, bridge, kernel, mapping)
    model.load_state_dict(state, strict=True, assign=True)
    for module in model.modules():
        if hasattr(module, "inv_freq") and module.inv_freq.is_meta:
            replacement = type(module)(config, device="cpu")
            module.register_buffer("inv_freq", replacement.inv_freq, persistent=False)
            module.original_inv_freq = replacement.original_inv_freq
    if config.tie_word_embeddings:
        raise ValueError("CPU embedding adapter requires untied weights")
    for name, module in model.model.named_children():
        if name != "embed_tokens":
            module.cuda()
    model.lm_head.cuda()
    model.model.embed_tokens.register_forward_pre_hook(lambda module, args: (args[0].cpu(),))
    model.model.embed_tokens.register_forward_hook(lambda module, args, value: value.cuda())
    model.register_parameter(
        "_generation_device", torch.nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False)
    )
    return model.eval()
