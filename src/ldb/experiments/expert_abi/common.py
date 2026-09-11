"""Strict artifacts and document-level metrics for the expert ABI study."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20260911


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze(root, names, output):
    write(root / output, {str(p): sha(root / p) for p in names})


def verify(root, name):
    for filename, expected in read(root / name).items():
        if sha(root / filename) != expected:
            raise ValueError(f"Frozen artifact changed: {filename}")


def setup(root):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    verify(root, "input-freeze.json")
    verify(root, "source-freeze.json")
    for archived, expected in read(root / "source-freeze.json").items():
        current = Path(archived).relative_to("sources")
        if sha(current) != expected:
            raise ValueError(f"Execution source differs from snapshot: {current}")
    return read(root / "models.json")


def load(record):
    model, info = AutoModelForCausalLM.from_pretrained(
        record["path"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        experts_implementation="eager",
        output_loading_info=True,
    )
    if any(info[k] for k in ["missing_keys", "unexpected_keys", "mismatched_keys"]):
        raise ValueError(f"Incomplete model load: {info}")
    return model.eval()


def tokenizer(root, index):
    tok = AutoTokenizer.from_pretrained(root / f"tokenizer-{index}", local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def batches(root, index, split):
    tok = tokenizer(root, index)
    texts = read(root / f"{split}.json")["texts"]
    return [
        tok(texts[i : i + 2], padding=True, truncation=True, max_length=128, return_tensors="pt")
        for i in range(0, len(texts), 2)
    ]


def interval(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite document measurements")
    draws = np.random.default_rng(SEED).integers(len(values), size=(10000, len(values)))
    return np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist()


def summarize(rows, originals):
    delta = [a["nll"] - b["nll"] for a, b in zip(rows, originals, strict=True)]
    ci = interval(delta)
    kl = float(np.mean([r["kl"] for r in rows]))
    return dict(
        delta_nll=float(np.mean(delta)),
        nll_interval=ci,
        mean_kl=kl,
        kl_interval=interval([r["kl"] for r in rows]),
        agreement=float(np.mean([r["agreement"] for r in rows])),
        quality_pass=bool(np.mean(delta) <= 0.03 and ci[1] <= 0.05 and kl <= 0.01),
    )


def aligned_layers(na, nb):
    """Monotone unique depth matches, defined independently of measured outcomes."""
    n = min(na, nb)
    return list(
        zip(
            np.rint(np.linspace(0, na - 1, n)).astype(int).tolist(),
            np.rint(np.linspace(0, nb - 1, n)).astype(int).tolist(),
            strict=True,
        )
    )
