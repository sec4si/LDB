"""Convert CPU-resident Qwen routed experts one matrix at a time on CUDA."""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .helpers import sha, storage_bytes, write
from .mxmoe_bridge import Bridge
from .mxmoe_model import PaperBank


@torch.inference_mode()
def run(
    source,
    library,
    output,
    rotation_source=None,
    allocation=None,
    activation_mixed=False,
    save_only=False,
    linear_mixed=False,
):
    if linear_mixed and activation_mixed:
        raise ValueError("Choose one mixed granularity")
    manifest = source / "download-complete.json"
    if not manifest.exists():
        raise ValueError("Weight download is not complete")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    write(
        output / "freeze.json",
        dict(
            download_manifest_sha256=sha(manifest),
            library_sha256=sha(library),
            sources={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            residual_rotation=rotation_source is not None,
            rotation_source_sha256=sha(rotation_source) if rotation_source else None,
            allocation_sha256=sha(allocation) if allocation else None,
            activation_mixed=activation_mixed,
            linear_mixed=linear_mixed,
            save_only=save_only,
            scope="RTN routed-expert control; default W4 or explicit 4/8-bit allocation; other weights FP16; no GPTQ or paper-quality claim",
        ),
    )
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    layers = [
        (i, layer_row)
        for i, layer_row in enumerate(model.model.layers)
        if hasattr(layer_row.mlp, "experts")
    ]
    plan = (
        json.loads(allocation.read_text())["plan"]
        if allocation
        else [[4] * layer_row.mlp.experts.num_experts for _, layer_row in layers]
    )
    if (activation_mixed or linear_mixed) and allocation is None:
        raise ValueError("Mixed activation requires an explicit format plan")
    if len(plan) != len(layers):
        raise ValueError("Incomplete layer plan")
    for (_, layer), choices in zip(layers, plan, strict=True):
        if len(choices) != layer.mlp.experts.num_experts:
            raise ValueError("Incomplete expert plan")
        for choice in choices:
            parts = ("gate", "up", "down") if linear_mixed else ("gu", "down")
            bits = [choice[part] for part in parts] if isinstance(choice, dict) else [choice]
            valid = ("w4a16", "w8a8") if activation_mixed or linear_mixed else (4, 8)
            if any(b not in valid for b in bits):
                raise ValueError("Unsupported precision in plan")
    if rotation_source is not None:
        from .mxmoe_rotation import rotate_qwen_residual

        rotate_qwen_residual(model, rotation_source)
    model.config.save_pretrained(output)
    kind = (
        "linear_mixed"
        if linear_mixed
        else "activation_mixed"
        if activation_mixed
        else "weight_only"
    )
    write(output / "allocation.json", dict(plan=plan, format_kind=kind))
    bridge = Bridge(library)
    kernel = next(
        i for i, n in enumerate(bridge.names) if n.endswith("groupgemm_hz_fused_0_cached")
    )
    for (li, layer), choices in zip(layers, plan, strict=True):
        old = layer.mlp.experts
        if linear_mixed:
            from .mxmoe_linear_bank import LinearBank

            bank = LinearBank(old.cuda(), choices, bridge, kernel)
        elif activation_mixed:
            from .mxmoe_activation_bank import ActivationBank

            bank = ActivationBank(old.cuda(), choices, bridge, kernel)
        else:
            bank = PaperBank(old, choices, bridge, kernel, device="cuda")
        bank.grouped_routing = True
        bank.reuse_prepared = not (activation_mixed or linear_mixed)
        if save_only:
            bank.cpu()
        layer.mlp.experts = bank
        del old
        gc.collect()
        print("packed layer", li, "cuda bytes", torch.cuda.memory_allocated(), flush=True)
    if save_only:
        write(
            output / "build.json",
            dict(
                storage_bytes=storage_bytes([model]),
                scope="CPU checkpoint conversion only; generation not tested",
            ),
        )
    else:
        model.cuda()
        torch.cuda.synchronize()
        tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
        inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs, max_new_tokens=16, do_sample=False, use_cache=True, output_router_logits=False
        )
        write(
            output / "smoke.json",
            dict(
                tokens=out.tolist(),
                text=tok.decode(out[0]),
                storage_bytes=storage_bytes([model]),
                peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                scope="Generation smoke only; not a quality or speed benchmark",
            ),
        )
    torch.save(model.state_dict(), output / "state.pt")
    print("saved", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "library", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rotation-source", type=Path)
    p.add_argument("--allocation", type=Path)
    p.add_argument("--activation-mixed", action="store_true")
    p.add_argument("--linear-mixed", action="store_true")
    p.add_argument("--save-only", action="store_true")
    run(**vars(p.parse_args()))
