"""Full-model document metrics and baseline replay without retained logits in RAM."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from .common import batches, freeze, load, read, setup, sha, summarize, verify, write
from .interfaces import KINDS, forward, load_maps, mapping
from .model import body, replace, weights, zero_layer


def evaluate(model, bb, baseline, create=False, replay=False):
    if create:
        baseline.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.inference_mode():
        for k, batch in enumerate(bb):
            logits = (
                model(**batch, use_cache=False, output_router_logits=False)
                .logits[:, :-1]
                .contiguous()
            )
            filename = baseline / f"{k}.npy"
            if create:
                np.save(filename, logits.numpy())
                original = logits
            else:
                original = torch.from_numpy(np.load(filename))
            if replay:
                if not torch.equal(logits, original):
                    raise RuntimeError(f"Baseline logits not restored: {filename}")
                continue
            lp, op = (
                torch.log_softmax(logits.double(), -1),
                torch.log_softmax(original.double(), -1),
            )
            for i, valid in enumerate(batch["attention_mask"][:, 1:].bool()):
                b, a = lp[i][valid], op[i][valid]
                labels = batch["input_ids"][i, 1:][valid]
                rows.append(
                    dict(
                        nll=float(-b.gather(1, labels[:, None]).mean()),
                        kl=float((a.exp() * (a - b)).sum(1).mean()),
                        accuracy=float((b.argmax(1) == labels).double().mean()),
                        agreement=float((a.argmax(1) == b.argmax(1)).double().mean()),
                        tokens=int(valid.sum()),
                    )
                )
    return rows


def baseline_for(root, model, m, split, bb):
    path = root / "baselines" / f"model-{m}" / split
    record = path / "documents.json"
    if record.exists():
        return path, read(record)
    rows = evaluate(model, bb, path, create=True)
    write(record, rows)
    return path, rows


def layer_effects(root):
    records = setup(root)
    verify(root, "capture-freeze.json")
    results = []
    for m, record in enumerate(records):
        model = load(record)
        for split in ["selection", "heldout"]:
            bb = batches(root, m, split)
            baseline, original = baseline_for(root, model, m, split, bb)
            for layer in range(len(model.model.layers)):
                with zero_layer(model, layer):
                    rows = evaluate(model, bb, baseline)
                results.append(
                    dict(
                        model=m,
                        layer=layer,
                        split=split,
                        **summarize(rows, original),
                        documents=rows,
                        originals=original,
                    )
                )
                write(root / "layer-effects-partial.json", results)
                print(
                    "Layer ablation", m, split, layer, round(results[-1]["mean_kl"], 5), flush=True
                )
            evaluate(model, bb, baseline, replay=True)
        del model
        gc.collect()
    write(root / "layer-effects.json", results)
    freeze(root, ["layer-effects.json"], "layer-effect-freeze.json")


def functions(root, records, recipient, donor, pairs, side, variant):
    other = "b" if side == "a" else "a"
    maps = None
    if variant in KINDS or variant in ["random", "predictor"]:
        kind = variant if variant in KINDS else "rank32"
        maps = load_maps(root / "interfaces" / f"{recipient}-from-{donor}-{kind}.safetensors")
    replacement = []
    for pair in pairs:
        layer, expert = pair[f"{side}_layer"], pair[f"{side}_expert"]
        dl, de = pair[f"{other}_layer"], pair[f"{other}_expert"]
        if variant == "same_slot":
            # The depth-matched layer is retained; expert ID is copied from the recipient.
            de = expert
            config = read(Path(records[donor]["path"]) / "config.json")
            if de >= config["num_local_experts"]:
                raise ValueError("Same-index donor unavailable")
        if variant == "zero":
            replacement.append((layer, expert, None))
            continue
        if variant == "predictor":
            replacement.append((layer, expert, lambda x, maps=maps: mapping(x, maps["predictor"])))
            continue
        payload = weights(records[donor], dl, de)
        if variant == "random":
            generator = torch.Generator().manual_seed(20260911 + 10000 * donor + 100 * dl + de)
            payload = tuple(torch.randn(p.shape, generator=generator) * p.std() for p in payload)
        if variant in ["raw", "same_slot"]:

            def function(x, payload=payload):
                return body(x, payload)
        else:

            def function(x, payload=payload, maps=maps):
                return forward(x, payload, maps)

        replacement.append((layer, expert, function))
    return replacement


def exchanges(root):
    records = setup(root)
    verify(root, "interface-freeze.json")
    verify(root, "selection-freeze.json")
    choices = read(root / "choices.json")
    results, replayed = [], []
    for recipient, record in enumerate(records):
        model = load(record)
        bb = batches(root, recipient, "heldout")
        baseline, originals = baseline_for(root, model, recipient, "heldout", bb)
        for choice in choices:
            if recipient not in [choice["a"], choice["b"]]:
                continue
            side = "a" if recipient == choice["a"] else "b"
            donor = choice["b"] if side == "a" else choice["a"]
            selected = choice["selected"]
            groups = [
                (f"single-{i}", [p], ["zero", "raw", "rank32"]) for i, p in enumerate(selected)
            ]
            groups += [
                (
                    f"joint-{n}",
                    selected[:n],
                    ["zero", "raw", *KINDS, "random", "predictor", "same_slot"],
                )
                for n in [2, 4]
            ]
            for group, pairs, variants in groups:
                for variant in variants:
                    start = time.monotonic()
                    if variant == "same_slot":
                        count = read(Path(records[donor]["path"]) / "config.json")[
                            "num_local_experts"
                        ]
                        if any(p[f"{side}_expert"] >= count for p in pairs):
                            results.append(
                                dict(
                                    recipient=recipient,
                                    donor=donor,
                                    group=group,
                                    variant=variant,
                                    unavailable="Recipient expert index absent in donor",
                                    pairs=pairs,
                                )
                            )
                            continue
                    replacements = functions(root, records, recipient, donor, pairs, side, variant)
                    with replace(model, replacements):
                        rows = evaluate(model, bb, baseline)
                    result = dict(
                        recipient=recipient,
                        donor=donor,
                        group=group,
                        variant=variant,
                        pairs=pairs,
                        **summarize(rows, originals),
                        documents=rows,
                        originals=originals,
                        seconds=time.monotonic() - start,
                    )
                    results.append(result)
                    write(root / "exchange-partial.json", results)
                    print(
                        "Exchange",
                        recipient,
                        "<-",
                        donor,
                        group,
                        variant,
                        round(result["delta_nll"], 6),
                        round(result["mean_kl"], 6),
                        result["quality_pass"],
                        flush=True,
                    )
                evaluate(model, bb, baseline, replay=True)
                replayed.append(
                    dict(recipient=recipient, donor=donor, group=group, bitwise_logits_equal=True)
                )
                write(root / "rollback-partial.json", replayed)
        del model
        gc.collect()
    hashes = []
    for record in records:
        actual = sha(Path(record["path"]) / "model.safetensors")
        if actual != record["weight_sha256"]:
            raise RuntimeError("Source checkpoint changed")
        hashes.append(actual)
    write(root / "exchanges.json", results)
    write(root / "rollback.json", dict(groups=replayed, unchanged_checkpoint_hashes=hashes))
    freeze(root, ["exchanges.json", "rollback.json"], "exchange-freeze.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["layers", "exchanges"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    (layer_effects if args.stage == "layers" else exchanges)(args.root)
