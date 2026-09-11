"""Sequential Qwen GPTQ with CPU layer offload and exact upstream code packing."""

import argparse
import contextlib
import gc
import io
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from .helpers import sha, write
from .mxmoe_bridge import Bridge
from .mxmoe_gptq_probe import upstream_definitions
from .mxmoe_model import PaperBank
from .mxmoe_rotation import hadamard_matrix, rotate_qwen_residual
from .mxmoe_screen import replace


def forward_layer(model, layer, state):
    x = state.cuda()
    positions = torch.arange(x.shape[1], device="cuda")[None]
    mask = create_causal_mask(
        config=model.config,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=positions,
    )
    return layer(
        x,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        position_embeddings=model.model.rotary_emb(x, positions),
    ).cpu()


def quantize(weight, x, bits, api, bridge):
    linear = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, device="cuda", dtype=torch.float32
    )
    linear.weight.copy_(weight)
    solver = api["GPTQ"](linear)
    solver.quantizer = api["Quantizer"]()
    solver.quantizer.configure(bits, perchannel=True, sym=False, mse=False)
    solver.add_batch(x, None)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.fasterquant(percdamp=0.01, groupsize=-1, actorder=True, static_groups=True)
    q = solver.quantizer
    codes = torch.round(linear.weight / q.scale + q.zero).to(torch.int32).contiguous()
    params = torch.cat([q.scale, -q.scale * q.zero], dim=1).half().contiguous()
    return bridge.pack_codes(codes, params, bits)


@torch.inference_mode()
def run(
    source,
    corpus,
    allocation,
    library,
    gptq_source,
    output,
    count=32,
    sequence_length=128,
    down_hadamard=False,
    rotation_seed=42,
):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads(allocation.read_text())["plan"]
    data = json.loads(corpus.read_text())
    samples = data.get("input_ids", data.get("texts", []))[:count]
    if len(samples) != count:
        raise ValueError("Insufficient calibration documents")
    had_source = gptq_source.with_name("hadamard_utils.py")
    write(
        output / "freeze.json",
        dict(
            library_sha256=sha(library),
            allocation_sha256=sha(allocation),
            corpus_sha256=sha(corpus),
            count=count,
            sequence_length=sequence_length,
            down_hadamard=down_hadamard,
            rotation_seed=rotation_seed,
            gptq_sha256=sha(gptq_source),
            hadamard_sha256=sha(had_source),
            source_manifest_sha256=sha(source / "download-complete.json"),
            sources={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            scope="Sequential GPTQ and residual rotation; optional online routed-down Hadamard; other weights FP16; fixed allocation, not full paper selector",
        ),
    )
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    if len(plan) != len(model.model.layers) or any(
        not hasattr(layer_row.mlp, "experts") for layer_row in model.model.layers
    ):
        raise ValueError("Requires a complete all-sparse Qwen layer plan")
    if set(model.config.layer_types) != {"full_attention"}:
        raise ValueError("Requires full attention")
    rotate_qwen_residual(model, had_source, seed=rotation_seed)
    model.config.save_pretrained(output)
    write(output / "allocation.json", dict(plan=plan))
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if "input_ids" in data:
        ids = [torch.tensor(row, dtype=torch.long) for row in samples]
        if any(tuple(x.shape) != (1, sequence_length) for x in ids):
            raise ValueError("Token corpus sequence length mismatch")
    else:
        ids = [
            tok(t, truncation=True, max_length=sequence_length, return_tensors="pt").input_ids
            for t in samples
        ]
    write(output / "calibration-inputs.json", dict(input_ids=[x.tolist() for x in ids]))
    model.model.embed_tokens.cuda()
    states = [model.model.embed_tokens(x.cuda()).cpu() for x in ids]
    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cuda()
    bridge = Bridge(library)
    api = upstream_definitions(gptq_source)
    kernel = next(
        i for i, n in enumerate(bridge.names) if n.endswith("groupgemm_hz_fused_0_cached")
    )
    rotation = (
        hadamard_matrix(had_source, model.config.moe_intermediate_size, "cuda").float()
        if down_hadamard
        else None
    )
    records = []
    for li, layer in enumerate(model.model.layers):
        layer.cuda()
        captured = []

        def capture(module, args, captured=captured):
            x, idx, _ = args
            captured.append((x.cpu(), idx.cpu()))

        handle = layer.mlp.experts.register_forward_pre_hook(capture)
        for state in states:
            forward_layer(model, layer, state)
        handle.remove()
        old = layer.mlp.experts
        bank = PaperBank(old, plan[li], bridge, kernel, rotation)
        for e, choice in enumerate(plan[li]):
            x = (
                torch.cat([value[torch.where(idx == e)[0]] for value, idx in captured])
                .cuda()
                .half()
            )
            routes = len(x)
            if routes:
                gu_bits = choice["gu"] if isinstance(choice, dict) else choice
                down_bits = choice["down"] if isinstance(choice, dict) else choice
                gu = quantize(old.gate_up_proj[e], x, gu_bits, api, bridge)
                gate, up = bridge.run([gu], [x], kernel)[0].chunk(2, -1)
                hidden = nn.functional.silu(gate) * up
                weight = old.down_proj[e]
                if rotation is not None:
                    hidden = hidden.float() @ rotation
                    weight = weight.float() @ rotation
                down = quantize(weight, hidden.half().contiguous(), down_bits, api, bridge)
                replace(bank, e, [gu, down])
            records.append(
                dict(
                    layer=li, expert=e, routes=routes, method="GPTQ" if routes else "RTN-no-routes"
                )
            )
            print("calibrated", li, e, "routes", routes, flush=True)
        layer.mlp.experts = bank
        bank.grouped_routing = True
        states = [forward_layer(model, layer, state) for state in states]
        layer.cpu()
        del old, captured, x
        torch.save(bank.state_dict(), output / f"layer-{li}.pt")
        write(output / "calibration.json", records)
        gc.collect()
        torch.cuda.empty_cache()
    model.model.norm.cuda()
    model.lm_head.cuda()
    logits = model.lm_head(model.model.norm(states[0].cuda())).cpu()
    torch.save(dict(input_ids=ids[0], logits=logits), output / "validation.pt")
    model.cpu()
    torch.save(model.state_dict(), output / "state.pt")
    print("saved", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "corpus", "allocation", "library", "gptq-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--down-hadamard", action="store_true")
    p.add_argument("--rotation-seed", type=int, default=42)
    run(**vars(p.parse_args()))
