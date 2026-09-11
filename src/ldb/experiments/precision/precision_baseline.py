"""Fresh development corpus and uniform precision controls for a single MoE."""

import gc
import json
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .expert_precision import convert
from .helpers import evaluate, load, sha, split_batches, storage_bytes, summarize, texts_in, write

ROOT = Path("runs/expert-mixed-precision")
MODEL = Path("models/base")


def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    tok = tokenizer()
    prior, sources = set(), {}
    for base in (Path("runs"), Path("docs/assets")):
        for path in sorted(base.rglob("*.json")):
            if path.is_relative_to(ROOT):
                continue
            found = set(texts_in(json.loads(path.read_text())))
            if found:
                prior.update(found)
                sources[str(path)] = sha(path)
    prefixes = {tuple(tok.encode(t, truncation=True, max_length=128)) for t in prior}
    texts, indices, skipped = [], [], []
    offset = 72000
    while len(texts) < 32:
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu"
            f"&config=sample-10BT&split=train&offset={offset}&length=32"
        )
        with urllib.request.urlopen(url, timeout=45) as response:
            rows = json.load(response)["rows"]
        for row in rows:
            text = row["row"]["text"]
            key = tuple(tok.encode(text, truncation=True, max_length=128))
            if text in prior or key in prefixes:
                skipped.append(row["row_idx"])
                continue
            texts.append(text)
            indices.append(row["row_idx"])
            prior.add(text)
            prefixes.add(key)
            if len(texts) == 32:
                break
        offset += 32
    write(ROOT / "development.json", dict(texts=texts, row_indices=indices, skipped=skipped))
    paths = [
        Path(__file__),
        Path("experiments/engineering_trials/expert_precision.py"),
        Path("docs/expert-mixed-precision-protocol.md"),
        Path("experiments/engineering_trials/helpers.py"),
        Path("experiments/engineering_trials/helpers.py"),
        MODEL / "model.safetensors",
        MODEL / "config.json",
        ROOT / "development.json",
    ]
    paths.extend(MODEL.glob("*token*.json"))
    write(
        ROOT / "input-freeze.json",
        dict(
            files={str(p): sha(p) for p in paths},
            prior_text_sources=sources,
            previous_unique_texts=len(prior) - 32,
            model="FlameF0X/TinyMoE-100m-2x8-retrained",
            max_length=128,
            batch_size=2,
        ),
    )


def verify():
    frozen = json.loads((ROOT / "input-freeze.json").read_text())
    for name, digest in frozen["files"].items():
        assert sha(Path(name)) == digest, name


def run():
    torch.set_num_threads(4)
    if not ROOT.exists():
        prepare()
    verify()
    output = ROOT / "baseline.json"
    if output.exists():
        raise FileExistsError(output)
    batches = split_batches(
        tokenizer(), json.loads((ROOT / "development.json").read_text())["texts"], 128
    )
    model = load(MODEL)
    baseline, original = evaluate(model, batches)
    native_bytes = storage_bytes([model])
    del model
    gc.collect()
    results = []
    for precision in ("int4", "int8", "bf16"):
        model = load(MODEL)
        plan = [[precision] * model.config.num_local_experts for _ in model.model.layers]
        convert(model, plan)
        _, documents = evaluate(model, batches, baseline)
        row = dict(
            precision=precision,
            documents=documents,
            tensor_bytes=storage_bytes([model]),
            **summarize(documents, original),
        )
        results.append(row)
        write(ROOT / f"baseline-{precision}.json", row)
        print(precision, row["tensor_bytes"], row["mean_kl"], row["passes_quality"], flush=True)
        del model
        gc.collect()
    write(
        output,
        dict(
            original=original,
            native_tensor_bytes=native_bytes,
            results=results,
            input_freeze_sha256=sha(ROOT / "input-freeze.json"),
        ),
    )


if __name__ == "__main__":
    run()
