"""Stream exact routed-block perturbations through native Qwen calibration layers."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from .helpers import sha, write
from .mxmoe_activation_probe import pack8, prepare
from .mxmoe_bridge import Bridge
from .qwen_expert_replay import (
    capture,
    layer_output,
    ordered_contributions,
    perturbed_layer_rows,
    replay,
)
from .qwen_gptq_build import forward_layer


def candidate_matrix(bridge, weight, fmt):
    if fmt == "w4a16":
        return bridge.pack(weight.contiguous(), 4)[0]
    packed, scale, _ = pack8(weight.contiguous())
    return packed, scale, weight.shape[0], weight.shape[1], 8


def candidate_contribution(experts, x, indices, weights, expert, part, fmt, matrix, bridge, kernel):
    slot, token = torch.where(indices.T == expert)
    selected = x[token]
    if not token.numel():
        return token, selected

    def linear(value):
        if fmt == "w8a8":
            packed, scale, _ = pack8(value)
            abits = 8
        else:
            packed, scale, abits = value, matrix[1], 16
        return prepare(bridge, kernel, [matrix], [packed], [scale], [abits], coalesce=True)()[0]

    gu = linear(selected) if part == "gu" else F.linear(selected, experts.gate_up_proj[expert])
    gate, up = gu.chunk(2, -1)
    if part == "gate":
        gate = linear(selected)
    if part == "up":
        up = linear(selected)
    hidden = experts.act_fn(gate) * up
    value = linear(hidden) if part == "down" else F.linear(hidden, experts.down_proj[expert])
    return token, (value * weights[token, slot, None]).to(x.dtype)


@torch.inference_mode()
def run(source, corpus, library, output, windows=128, layers=24, kernel=0, linear_mixed=False):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ids = json.loads(corpus.read_text())["input_ids"]
    if not 1 <= windows <= len(ids):
        raise ValueError("Invalid window count")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    if not 1 <= layers <= len(model.model.layers):
        raise ValueError("Invalid layer count")
    bridge = Bridge(library)
    write(
        output / "freeze.json",
        dict(
            corpus_sha256=sha(corpus),
            library_sha256=sha(library),
            model_manifest_sha256=sha(source / "download-complete.json"),
            windows=windows,
            layers=layers,
            kernel=kernel,
            kernel_name=bridge.names[kernel],
            linear_mixed=linear_mixed,
            sources={
                name: sha(Path(__file__).parent / name)
                for name in (
                    "qwen_mixed_loss_profile.py",
                    "qwen_expert_replay.py",
                    "qwen_gptq_build.py",
                    "mxmoe_activation_probe.py",
                    "mxmoe_bridge.py",
                )
            },
            scope="Native unrotated inputs; actual RTN W4A16/W8A8 kernels; isolated perturbations; no GPTQ or allocation quality claim",
        ),
    )
    model.model.embed_tokens.cuda()
    states = [model.model.embed_tokens(torch.tensor(x, device="cuda")).cpu() for x in ids[:windows]]
    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cuda()
    for li, layer in enumerate(model.model.layers[:layers]):
        layer.cuda()
        experts = layer.mlp.experts
        directory = output / f"layer-{li}"
        directory.mkdir()

        def blocks(e):
            gate, up = experts.gate_up_proj[e].chunk(2, 0)
            return (
                [("gate", gate), ("up", up), ("down", experts.down_proj[e])]
                if linear_mixed
                else [("gu", experts.gate_up_proj[e]), ("down", experts.down_proj[e])]
            )

        matrices = {
            (e, p, f): candidate_matrix(bridge, w, f)
            for e in range(experts.num_experts)
            for p, w in blocks(e)
            for f in ("w4a16", "w8a8")
        }
        sums = {key: 0.0 for key in matrices}
        checks = []
        next_states = []
        for wi, state in enumerate(states):
            saved = {}

            def residual_hook(module, args, saved=saved):
                saved["residual"] = args[0]

            def expert_hook(module, args, saved=saved):
                saved["args"] = args

            hooks = [
                layer.post_attention_layernorm.register_forward_pre_hook(residual_hook),
                experts.register_forward_pre_hook(expert_hook),
            ]
            try:
                baseline = forward_layer(model, layer, state)
            finally:
                for hook in hooks:
                    hook.remove()
            next_states.append(baseline)
            x, indices, weights = saved["args"]
            residual = saved["residual"]
            shared = torch.sigmoid(layer.mlp.shared_expert_gate(x)) * layer.mlp.shared_expert(x)
            contributions = capture(experts, x, indices, weights)
            ordered = ordered_contributions(x, indices, contributions)
            assert torch.equal(
                layer_output(replay(x, contributions), shared, residual).cpu(), baseline
            )
            base = baseline.cuda().reshape_as(x)
            scores = []
            for (e, p, f), matrix in matrices.items():
                changed = candidate_contribution(
                    experts, x, indices, weights, e, p, f, matrix, bridge, kernel
                )
                token, value = perturbed_layer_rows(ordered, indices, e, changed, shared, residual)
                squared = (value.double() - base[token].double()).square().sum().item()
                if not math.isfinite(squared):
                    raise ValueError((li, wi, e, p, f, "Nonfinite loss"))
                sums[e, p, f] += squared
                scores.append(dict(expert=e, part=p, format=f, squared_error=squared))
                # Independently rerun all experts and the complete decoder layer.
                if wi == 0 and e in (0, 17, 59):
                    original_forward = experts.forward

                    def direct_forward(xx, ii, ww):
                        cc = capture(experts, xx, ii, ww)
                        cc[e] = candidate_contribution(
                            experts, xx, ii, ww, e, p, f, matrix, bridge, kernel
                        )
                        return replay(xx, cc)

                    experts.forward = direct_forward
                    try:
                        direct = forward_layer(model, layer, state)
                    finally:
                        experts.forward = original_forward
                    reconstructed = baseline.reshape_as(x).clone()
                    reconstructed[token.cpu()] = value.cpu()
                    assert torch.equal(direct, reconstructed.reshape_as(baseline)), (li, e, p, f)
                    norm = float(torch.linalg.vector_norm(direct.double() - baseline.double()))
                    assert math.isclose(norm, math.sqrt(squared), rel_tol=1e-12, abs_tol=1e-12)
                    checks.append(dict(expert=e, part=p, format=f, exact=True, direct_norm=norm))
            write(directory / f"window-{wi}.json", scores)
            print("profiled layer", li, "window", wi, flush=True)
        rows = [
            dict(expert=e, part=p, format=f, loss=math.sqrt(v), squared_error=v)
            for (e, p, f), v in sums.items()
        ]
        write(
            directory / "result.json",
            dict(
                windows=windows,
                rows=rows,
                checks=checks,
                aggregation="sqrt(sum of FP64 squared errors over all affected rows and windows)",
            ),
        )
        states = next_states
        del matrices, contributions, ordered, base, shared, saved, x, residual, changed, value
        layer.cpu()
        torch.cuda.empty_cache()
    write(
        output / "complete.json",
        dict(layers=layers, windows=windows, candidates_per_layer=360 if linear_mixed else 240),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "corpus", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--kernel", type=int, default=0)
    parser.add_argument("--linear-mixed", action="store_true")
    run(**vars(parser.parse_args()))
