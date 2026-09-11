"""Same-checkpoint paired generation timing for fused activation packing."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .helpers import interval, sha, write
from .mxmoe_activation_bank import ActivationBank
from .mxmoe_qwen_restore import restore


def place_with_cpu_embedding(model):
    if model.config.tie_word_embeddings:
        raise ValueError("CPU embedding placement requires an untied LM head")
    for name, module in model.model.named_children():
        if name != "embed_tokens":
            module.cuda()
    model.lm_head.cuda()
    embedding = model.model.embed_tokens
    if embedding.weight.device.type != "cpu":
        raise ValueError("Expected CPU checkpoint before placement")
    embedding.register_forward_pre_hook(lambda module, args: (args[0].cpu(),))
    embedding.register_forward_hook(lambda module, args, value: value.cuda())
    # Generation infers its execution device from the first parameter.
    model.register_parameter(
        "_generation_device", torch.nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False)
    )
    return model


@torch.inference_mode()
def run(
    source,
    checkpoint,
    library,
    corpus,
    output,
    count=8,
    repeats=2,
    optimization="packing",
    cpu_embedding=False,
):
    if count < 2 or repeats < 1:
        raise ValueError("Require at least two prompts and one pair per prompt")
    texts = json.loads(corpus.read_text())["texts"]
    if len(texts) < count:
        raise ValueError("Insufficient corpus prompts")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = restore(checkpoint, library, device="cpu" if cpu_embedding else "cuda")
    if cpu_embedding:
        place_with_cpu_embedding(model)
    banks = [layer.mlp.experts for layer in model.model.layers]
    if not all(isinstance(bank, ActivationBank) for bank in banks):
        raise ValueError("Expected mixed activation checkpoint")

    def select(enabled):
        for bank in banks:
            bank.fused_activation_pack = True if optimization == "metadata" else enabled
            bank.coalesce_metadata = enabled and optimization in ("metadata", "combined")

    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    write(
        output / "freeze.json",
        dict(
            checkpoint_sha256=sha(checkpoint / "state.pt"),
            library_sha256=sha(library),
            corpus_sha256=sha(corpus),
            count=count,
            repeats=repeats,
            input_limit=128,
            new_tokens=16,
            batch=1,
            optimization=optimization,
            cpu_embedding=cpu_embedding,
            sources={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            scope="Same weights and grouped executor; selected optimization only. Metadata comparison keeps fused packing in both arms. Run with otherwise idle GPU. Includes prefill, decode and host overhead; excludes load and compilation.",
        ),
    )
    rng = random.Random(42)
    rows = []
    for prompt, text in enumerate(texts[:count]):
        ids = tok(text, truncation=True, max_length=128, return_tensors="pt")["input_ids"].cuda()
        kwargs = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=True,
            do_sample=False,
            min_new_tokens=16,
            max_new_tokens=16,
            pad_token_id=tok.eos_token_id,
        )
        for fused in (False, True):
            select(fused)
            model.generate(**kwargs)
        for repeat in range(repeats):
            order = [False, True]
            rng.shuffle(order)
            row = dict(prompt=prompt, repeat=repeat, order=order, seconds={}, tokens={})
            for fused in order:
                select(fused)
                torch.cuda.synchronize()
                start = time.perf_counter()
                tokens = model.generate(**kwargs)
                torch.cuda.synchronize()
                name = "fused" if fused else "original"
                row["seconds"][name] = time.perf_counter() - start
                row["tokens"][name] = tokens[:, ids.shape[1] :].tolist()
            rows.append(row)
            write(output / "partial.json", rows)
            if row["tokens"]["fused"] != row["tokens"]["original"]:
                raise AssertionError("Generation mismatch; paired speed claim invalid")
            print(prompt, repeat, row["seconds"], flush=True)
    # Bootstrap prompt means, not repeated observations treated as independent.
    means = [
        sum(1 - r["seconds"]["fused"] / r["seconds"]["original"] for r in rows if r["prompt"] == p)
        / repeats
        for p in range(count)
    ]
    write(
        output / "result.json",
        dict(
            rows=rows,
            prompt_mean_reductions=means,
            mean_fractional_latency_reduction=sum(means) / count,
            prompt_bootstrap_interval=interval(means),
            equal_generations=len(rows),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "checkpoint", "library", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--optimization", choices=("packing", "metadata", "combined"), default="packing"
    )
    parser.add_argument("--cpu-embedding", action="store_true")
    run(**vars(parser.parse_args()))
