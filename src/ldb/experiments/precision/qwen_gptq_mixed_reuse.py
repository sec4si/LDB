"""Reuse GPTQ W4 buffers and rotate/pack only selected RTN W8A8 blocks."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from .helpers import sha, write
from .mxmoe_activation_probe import pack8
from .mxmoe_rotation import hadamard_matrix


@torch.inference_mode()
def run(source, checkpoint, allocation, library, hadamard_source, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    freeze = json.loads((checkpoint / "freeze.json").read_text())
    assert freeze["down_hadamard"] is False
    assert freeze["hadamard_sha256"] == sha(hadamard_source)
    assert freeze["source_manifest_sha256"] == sha(source / "download-complete.json")
    assert all(
        c == 4
        for layer in json.loads((checkpoint / "allocation.json").read_text())["plan"]
        for c in layer
    )
    plan = json.loads(allocation.read_text())["plan"]
    assert len(plan) == 24 and all(len(layer) == 60 for layer in plan)
    assert all(
        set(c) == {"gu", "down"} and all(v in ("w4a16", "w8a8") for v in c.values())
        for layer in plan
        for c in layer
    )
    state = torch.load(checkpoint / "state.pt", map_location="cpu", mmap=True, weights_only=True)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        with safe_open(source / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).half().to(device="cuda", dtype=torch.float64)

    q = hadamard_matrix(hadamard_source, 2048, "cuda")
    rng = torch.Generator(device="cuda").manual_seed(freeze["rotation_seed"])
    q = (torch.randint(0, 2, (2048,), device="cuda", generator=rng) * 2 - 1)[:, None] * q
    rows = []
    for li, choices in enumerate(plan):
        prefix = f"model.layers.{li}"
        norm = weight(prefix + ".post_attention_layernorm.weight")
        router = ((weight(prefix + ".mlp.gate.weight") * norm) @ q).half().cpu()
        assert torch.equal(router, state[prefix + ".mlp.gate.weight"]), (
            "Rotation basis mismatch",
            li,
        )
        for e, c in enumerate(choices):
            for p, fmt in c.items():
                if fmt == "w4a16":
                    continue
                original = prefix + f".mlp.experts.{e}."
                if p == "gu":
                    w = torch.cat(
                        [weight(original + "gate_proj.weight"), weight(original + "up_proj.weight")]
                    )
                    rotated = ((w * norm) @ q).half()
                else:
                    rotated = (q.T @ weight(original + "down_proj.weight")).half()
                packed, scale, _ = pack8(rotated.contiguous())
                target = prefix + f".mlp.experts.{p}_{e}"
                state[target + "_packed"] = packed.cpu()
                state[target + "_scale"] = scale.cpu()
                rows.append(dict(layer=li, expert=e, part=p))
        print("assembled GPTQ mixed layer", li, flush=True)
    shutil.copy2(checkpoint / "config.json", output / "config.json")
    write(output / "allocation.json", dict(plan=plan, format_kind="activation_mixed"))
    write(
        output / "freeze.json",
        dict(
            library_sha256=sha(library),
            parent_checkpoint_sha256=sha(checkpoint / "state.pt"),
            parent_freeze=freeze,
            allocation_sha256=sha(allocation),
            script_sha256=sha(Path(__file__)),
            scope="Reused rotated GPTQ W4; selected rotated RTN W8A8; transferred unrotated RTN allocation, not recalibrated GPTQ allocation",
        ),
    )
    torch.save(state, output / "state.pt")
    write(
        output / "build.json",
        dict(
            upgrades=rows,
            state_tensor_bytes=sum(v.numel() * v.element_size() for v in state.values()),
            rotation_router_checks=24,
            scope="State tensor bytes exclude nonpersistent runtime buffers",
        ),
    )
    print("saved", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "checkpoint", "allocation", "library", "hadamard-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
