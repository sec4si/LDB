"""Capture FP16 expert routing on exact paper calibration token windows."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from .helpers import sha, write
from .qwen_gptq_build import forward_layer


@torch.inference_mode()
def run(source, corpus, output, last_layer=0):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    values = json.loads(corpus.read_text())["input_ids"]
    ids = [torch.tensor(x, dtype=torch.long) for x in values]
    if len(ids) != 128 or any(x.shape != (1, 4096) for x in ids):
        raise ValueError("Require exact 128 by 4096 calibration windows")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    if not 0 <= last_layer < len(model.model.layers):
        raise ValueError("Invalid final layer")
    write(
        output / "freeze.json",
        dict(
            source_manifest_sha256=sha(source / "download-complete.json"),
            corpus_sha256=sha(corpus),
            last_layer=last_layer,
            script_sha256=sha(Path(__file__)),
            scope="Native unrotated FP16 calibration routing; no loss scores or allocations yet",
        ),
    )
    model.model.embed_tokens.cuda()
    states = [model.model.embed_tokens(x.cuda()).cpu() for x in ids]
    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cuda()
    for li, layer in enumerate(model.model.layers[: last_layer + 1]):
        directory = output / f"layer-{li}"
        directory.mkdir()
        layer.cuda()
        count = 0

        def capture(module, args):
            nonlocal count
            torch.save(tuple(x.detach().cpu() for x in args), directory / f"{count}.pt")
            count += 1

        hook = layer.mlp.experts.register_forward_pre_hook(capture)
        states = [forward_layer(model, layer, state) for state in states]
        hook.remove()
        layer.cpu()
        assert count == 128
        write(
            directory / "complete.json",
            dict(
                windows=count,
                rows=128 * 4096,
                routed_assignments=128 * 4096 * 4,
                scope="Saved expert input, selected IDs and weights before any quantization",
            ),
        )
        print("captured native layer", li, flush=True)
    write(output / "complete.json", dict(layers=last_layer + 1, windows=128))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--last-layer", type=int, default=0)
    run(**vars(parser.parse_args()))
