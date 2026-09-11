"""Capture real mixed-model expert inputs for routed-bank replay."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .helpers import sha, write
from .mxmoe_qwen_restore import restore
from .qwen_gptq_build import forward_layer


@torch.inference_mode()
def run(source, checkpoint, library, corpus, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = restore(checkpoint, library, device="cpu")
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    texts = json.loads(corpus.read_text())["texts"][:8]
    if len(texts) != 8:
        raise ValueError("Need eight texts")
    ids = [
        tok(text, truncation=True, max_length=128, return_tensors="pt")["input_ids"]
        for text in texts
    ]
    model.model.embed_tokens.cuda()
    states = [model.model.embed_tokens(x.cuda()).cpu() for x in ids]
    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cuda()
    write(
        output / "freeze.json",
        dict(
            checkpoint_sha256=sha(checkpoint / "state.pt"),
            library_sha256=sha(library),
            corpus_sha256=sha(corpus),
            sources={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            scope="Real prefill expert inputs; last-row slicing replays a one-token bank workload, not KV-cache generation",
        ),
    )
    for li, layer in enumerate(model.model.layers):
        layer.cuda()
        records = []

        def capture(module, args):
            records.append(tuple(x.detach().cpu().clone() for x in args))

        hook = layer.mlp.experts.register_forward_pre_hook(capture) if li in (0, 11, 23) else None
        states = [forward_layer(model, layer, x) for x in states]
        if hook is not None:
            hook.remove()
            assert len(records) == 8
            torch.save(records, output / f"layer-{li}.pt")
        layer.cpu()
        print("captured through layer", li, flush=True)
    write(
        output / "result.json",
        dict(layers=[0, 11, 23], prompts=8, files={p.name: sha(p) for p in output.glob("*.pt")}),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "checkpoint", "library", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
