"""Fixed greedy continuations: inspect generation, without claiming a task benchmark."""

import argparse
from pathlib import Path

import torch

from .common import load, read, sha, tokenizer, write
from .evaluate import functions
from .model import replace

PROMPTS = [
    "The capital of France is",
    "The result of 2 + 3 is",
    "Water freezes at a temperature of",
    "def add(a, b):\n    return",
    "Once upon a time, a small bird",
    "Explain why the sky appears blue:",
]


def continuations(model, tok):
    output = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt")["input_ids"]
            prefix = ids.shape[1]
            for _ in range(24):
                logits = model(input_ids=ids, use_cache=False, output_router_logits=False).logits[
                    :, -1
                ]
                token = logits.argmax(-1, keepdim=True)
                ids = torch.cat([ids, token], 1)
                if int(token.item()) == tok.eos_token_id:
                    break
            output.append(
                dict(
                    prompt=prompt,
                    continuation=tok.decode(ids[0, prefix:], skip_special_tokens=True),
                    generated_token_ids=ids[0, prefix:].tolist(),
                )
            )
    return output


def run(root):
    torch.set_num_threads(4)
    write(
        root / "generation-protocol.json",
        dict(
            prompts=PROMPTS,
            max_new_tokens=24,
            greedy=True,
            raw_prompts=True,
            frozen_before_generation=True,
            source_sha256=sha(Path(__file__)),
            scope="Post-selection descriptive examples, not held-out task accuracy or interface selection data",
        ),
    )
    records = read(root / "models.json")
    output = []
    for m, record in enumerate(records):
        model, tok = load(record), tokenizer(root, m)
        baseline = continuations(model, tok)
        output.append(dict(model=m, variant="original", examples=baseline))
        for choice in read(root / "choices.json"):
            if m not in [choice["a"], choice["b"]]:
                continue
            side = "a" if m == choice["a"] else "b"
            donor = choice["b"] if side == "a" else choice["a"]
            for variant in ["raw", "rank32"]:
                with replace(
                    model, functions(root, records, m, donor, choice["selected"], side, variant)
                ):
                    rows = continuations(model, tok)
                output.append(dict(model=m, donor=donor, variant=variant, examples=rows))
                write(root / "generation-partial.json", output)
                print("Generated", m, donor, variant, flush=True)
        if continuations(model, tok) != baseline:
            raise RuntimeError("Generation rollback changed greedy tokens")
        del model
    write(root / "generation.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
