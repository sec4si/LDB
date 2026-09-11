"""Standalone evaluation and storage helpers extracted from the original studies."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load(path):
    model, info = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        experts_implementation="eager",
        output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    return model.eval()


def evaluate(model, batches, baseline=None):
    probs = []
    records = []
    with torch.inference_mode():
        for k, batch in enumerate(batches):
            lp = torch.log_softmax(
                model(**batch, use_cache=False, output_router_logits=False).logits[:, :-1].double(),
                dim=-1,
            )
            mask = batch["attention_mask"][:, 1:].bool()
            labels = batch["input_ids"][:, 1:]
            for i in range(len(lp)):
                b = lp[i][mask[i]]
                truth = labels[i][mask[i]]
                row = dict(
                    nll=float(-b.gather(1, truth[:, None]).mean()),
                    accuracy=float((b.argmax(1) == truth).double().mean()),
                )
                if baseline is not None:
                    a = baseline[k][i][mask[i]]
                    row.update(
                        kl=float((a.exp() * (a - b)).sum(1).mean()),
                        agreement=float((a.argmax(1) == b.argmax(1)).double().mean()),
                    )
                records.append(row)
            if baseline is None:
                probs.append(lp)
    return probs, records


def interval(values):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(20260910)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return list(map(float, np.quantile(means, [0.025, 0.975])))


def summarize(rows, original):
    dn = [a["nll"] - b["nll"] for a, b in zip(rows, original, strict=True)]
    kl = [a["kl"] for a in rows]
    ci = interval(dn)
    return dict(
        delta_nll=float(np.mean(dn)),
        nll_interval=ci,
        mean_kl=float(np.mean(kl)),
        kl_interval=interval(kl),
        passes_quality=bool(np.mean(dn) <= 0.03 and ci[1] <= 0.05 and np.mean(kl) <= 0.01),
    )


class PoolBank(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)

    def forward(self, x, indices, weights):
        result = torch.zeros_like(x)
        mask = nn.functional.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)
        for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten():
            slot, token = torch.where(mask[e])
            y = self.experts[int(e)](x[token]) * weights[token, slot, None]
            result.index_add_(0, token, y.to(result.dtype))
        return result


def storage_bytes(models):
    stores = {}
    for model in models:
        for tensor in list(model.parameters()) + list(model.buffers()):
            if not tensor.numel():
                continue
            store = tensor.untyped_storage()
            stores[store.data_ptr()] = store.nbytes()
    return sum(stores.values())


class StoredExpert(nn.Module):
    def __init__(self, gate_up, down):
        super().__init__()
        self.gu = gate_up
        self.down = down

    def forward(self, x):
        gate, up = self.gu(x).chunk(2, dim=-1)
        return self.down(nn.functional.silu(gate) * up)


def split_batches(tok, texts, length):
    return [
        tok(texts[i : i + 2], padding=True, truncation=True, max_length=length, return_tensors="pt")
        for i in range(0, len(texts), 2)
    ]


def texts_in(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "text" and isinstance(item, str):
                yield item
            elif key == "texts" and isinstance(item, list):
                yield from (x for x in item if isinstance(x, str))
            else:
                yield from texts_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from texts_in(item)
