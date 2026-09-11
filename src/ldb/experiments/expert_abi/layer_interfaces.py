"""Follow-up: layer-conditioned interfaces shared across unseen expert IDs."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import torch
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment

from .common import batches, freeze, load, read, setup, sha, summarize, tokenizer, verify, write
from .evaluate import baseline_for, evaluate
from .interfaces import KINDS, capture, forward, load_maps, mapping, pair_cost, save_maps
from .interfaces import fit as original_fit
from .model import body, replace, weights


def fit(x, y, kind, weight=None, identity=True):
    """Keep the objective fixed; use QR SVD if divide-and-conquer fails."""
    try:
        return original_fit(x, y, kind, weight, identity)
    except torch._C._LinAlgError:
        x, y = x.double(), y.double()
        w = torch.ones(len(x), dtype=torch.float64) if weight is None else weight.double().square()
        if not torch.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("Nonfinite or empty regression weights")
        w /= w.sum()
        mx, my = (w[:, None] * x).sum(0), (w[:, None] * y).sum(0)
        xc, yc = x - mx, y - my
        covariance = xc.T @ (w[:, None] * xc)
        ridge = max(float(covariance.trace()) / x.shape[1] * 0.001, 1e-12)
        target = yc - xc if identity else yc
        coefficient = torch.linalg.solve(
            covariance + ridge * torch.eye(x.shape[1], dtype=x.dtype), xc.T @ (w[:, None] * target)
        )
        if not torch.isfinite(coefficient).all():
            raise ValueError("Nonfinite regression coefficient")
        u, s, vh = [
            torch.from_numpy(v.copy())
            for v in svd(coefficient.numpy(), full_matrices=False, lapack_driver="gesvd")
        ]
        rank = int(kind.removeprefix("rank"))
        left, right = u[:, :rank] * s[:rank], vh[:rank]
        print("Numerical fallback: finite coefficient, scipy gesvd", kind, flush=True)
        return dict(
            left=left.float(),
            right=right.float(),
            bias=(my - mx @ left @ right - (mx if identity else 0)).float(),
            identity=torch.tensor(identity),
        )


def fit_all(parent, root):
    records = setup(parent)
    verify(parent, "selection-freeze.json")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(parent / "models.json", root / "models.json")
    shutil.copy2(parent / "choices.json", root / "choices.json")
    shutil.copy2("docs/expert-abi-layer-interface-protocol.md", root / "protocol.md")
    for m in range(3):
        shutil.copytree(parent / f"tokenizer-{m}", root / f"tokenizer-{m}")
    source_paths = sorted(Path(__file__).parent.relative_to(Path.cwd()).glob("*.py"))
    for path in source_paths:
        target = root / "sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    freeze(root, [str(Path("sources") / p) for p in source_paths], "source-freeze.json")
    directory = root / "interfaces"
    directory.mkdir()
    training = []
    for choice in read(root / "choices.json"):
        a, b = choice["a"], choice["b"]
        layers = sorted({(r["a_layer"], r["b_layer"]) for r in choice["selected"]})
        for la, lb in layers:
            ca, cb = capture(parent, a, "fit", la), capture(parent, b, "fit", lb)
            for kind in KINDS:
                native = {(a, b): (ca, cb, la), (b, a): (cb, ca, lb)}
                maps = {
                    direction: dict(H=fit(x["x"], y["x"], kind))
                    for direction, (x, y, _) in native.items()
                }
                for source, target in [(a, b), (b, a)]:
                    maps[source, target]["T"] = {
                        k: v.clone() for k, v in maps[target, source]["H"].items()
                    }
                    maps[source, target]["T"]["bias"].zero_()
                matrix, ea, eb, _, _ = pair_cost(
                    parent, records, a, b, (la, lb), "fit", maps[a, b], maps[b, a], 0
                )
                ii, jj = linear_sum_assignment(matrix)
                matched = [(ea[i], eb[j]) for i, j in zip(ii, jj, strict=True)]
                for source, target, x, sl, tl, pairs in [
                    (a, b, ca, la, lb, matched),
                    (b, a, cb, lb, la, [(f, e) for e, f in matched]),
                ]:
                    samples = []
                    for se, te in pairs:
                        with torch.no_grad():
                            z = body(
                                mapping(x["x"], maps[source, target]["H"]),
                                weights(records[target], tl, te),
                            )
                            y = body(x["x"], weights(records[source], sl, se))
                        samples.append((x["x"], z, y, x["route"][:, se]))
                    xx, zz, yy, ww = [torch.cat([r[i] for r in samples]) for i in range(4)]
                    maps[source, target]["T"] = fit(zz, yy, kind, ww)
                    if kind == "rank32":
                        maps[source, target]["predictor"] = fit(
                            xx, yy, "rank64", ww, identity=False
                        )
                    path = directory / f"{source}-from-{target}-layer-{sl}-{kind}.safetensors"
                    save_maps(path, maps[source, target])
                    parameters = {
                        name: sum(v.numel() for key, v in p.items() if key != "identity")
                        for name, p in maps[source, target].items()
                    }
                    training.append(
                        dict(
                            recipient=source,
                            donor=target,
                            recipient_layer=sl,
                            donor_layer=tl,
                            kind=kind,
                            training_expert_pairs=pairs,
                            parameters=parameters,
                        )
                    )
                print("Fitted per-layer shared maps", a, b, la, lb, kind, flush=True)
    write(root / "training.json", training)
    freeze(
        root,
        ["models.json", "choices.json", "training.json", "protocol.md"]
        + [str(p.relative_to(root)) for p in directory.glob("*")],
        "fit-freeze.json",
    )
    historical = set()
    for directory in [parent, Path("../expert-fingerprint-research/runs")]:
        for path in (
            directory.rglob("*.json") if directory == parent else directory.glob("*/*.json")
        ):
            if path.name not in [
                "corpus.json",
                "fit.json",
                "selection.json",
                "heldout.json",
                "calibration.json",
                "development.json",
            ]:
                continue
            data = read(path)
            if isinstance(data, dict):
                historical.update(t for t in data.get("texts", []) if isinstance(t, str))
    url = "https://datasets-server.huggingface.co/rows?" + urlencode(
        dict(
            dataset="HuggingFaceFW/fineweb-edu",
            config="sample-10BT",
            split="train",
            offset=1230000,
            length=64,
        )
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        texts = [r["row"]["text"] for r in json.load(response)["rows"]]
        revision = response.headers.get("x-revision")
    toks = [tokenizer(root, i) for i in range(3)]
    enc = [[tuple(t.encode(s, truncation=True, max_length=128)) for s in texts] for t in toks]
    prior_prefix = {tuple(toks[0].encode(s, truncation=True, max_length=128)) for s in historical}
    if (
        not revision
        or len(texts) != 64
        or len(set(texts)) != 64
        or set(texts) & historical
        or len(set(enc[0])) != 64
        or not enc[0] == enc[1] == enc[2]
        or set(enc[0]) & prior_prefix
    ):
        raise ValueError("Fresh confirmation data failed independence audit")
    write(root / "heldout.json", dict(texts=texts, revision=revision, url=url))
    write(
        root / "data-audit.json",
        dict(
            documents=64,
            native_encodings_equal=True,
            historical_exact_overlap=0,
            historical_prefix_overlap=0,
            revision=revision,
        ),
    )
    freeze(root, ["heldout.json", "data-audit.json"], "confirmation-freeze.json")


def evaluate_all(parent, root):
    records = setup(parent)
    for name in ["fit-freeze.json", "source-freeze.json", "confirmation-freeze.json"]:
        verify(root, name)
    for archived, expected in read(root / "source-freeze.json").items():
        if sha(Path(archived).relative_to("sources")) != expected:
            raise ValueError("Follow-up execution source changed")
    results, rollbacks = [], []
    for recipient, record in enumerate(records):
        model = load(record)
        bb = batches(root, recipient, "heldout")
        baseline, originals = baseline_for(root, model, recipient, "heldout", bb)
        for choice in read(root / "choices.json"):
            if recipient not in [choice["a"], choice["b"]]:
                continue
            side = "a" if recipient == choice["a"] else "b"
            other = "b" if side == "a" else "a"
            donor = choice[other]
            for variant in ["zero", "raw", "global-rank32", *KINDS, "random", "predictor"]:
                replacements = []
                for p in choice["selected"]:
                    layer, expert = p[f"{side}_layer"], p[f"{side}_expert"]
                    dl, de = p[f"{other}_layer"], p[f"{other}_expert"]
                    payload = weights(records[donor], dl, de)
                    if variant == "zero":
                        function = None
                    elif variant == "raw":
                        def function(x, payload=payload):
                            return body(x, payload)
                    else:
                        kind = variant if variant in KINDS else "rank32"
                        file = (
                            parent / "interfaces" / f"{recipient}-from-{donor}-rank32.safetensors"
                            if variant == "global-rank32"
                            else root
                            / "interfaces"
                            / f"{recipient}-from-{donor}-layer-{layer}-{kind}.safetensors"
                        )
                        maps = load_maps(file)
                        if variant == "predictor":
                            def function(x, maps=maps):
                                return mapping(x, maps["predictor"])
                        else:
                            if variant == "random":
                                generator = torch.Generator().manual_seed(
                                    20260911 + 10000 * donor + 100 * dl + de
                                )
                                payload = tuple(
                                    torch.randn(v.shape, generator=generator) * v.std()
                                    for v in payload
                                )
                            def function(x, payload=payload, maps=maps):
                                return forward(
                                                            x, payload, maps
                                                        )
                    replacements.append((layer, expert, function))
                with replace(model, replacements):
                    rows = evaluate(model, bb, baseline)
                result = dict(
                    recipient=recipient,
                    donor=donor,
                    variant=variant,
                    **summarize(rows, originals),
                    documents=rows,
                    originals=originals,
                )
                results.append(result)
                write(root / "partial.json", results)
                print(
                    "Layer-conditioned exchange",
                    recipient,
                    "<-",
                    donor,
                    variant,
                    round(result["delta_nll"], 6),
                    round(result["mean_kl"], 6),
                    result["quality_pass"],
                    flush=True,
                )
            evaluate(model, bb, baseline, replay=True)
            rollbacks.append(
                dict(recipient=recipient, donor=donor, complete_logits_bitwise_equal=True)
            )
        del model
    for record in records:
        if sha(Path(record["path"]) / "model.safetensors") != record["weight_sha256"]:
            raise ValueError("Source checkpoint changed")
    write(root / "results.json", results)
    write(root / "rollback.json", rollbacks)
    freeze(root, ["results.json", "rollback.json"], "result-freeze.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (fit_all if args.stage == "fit" else evaluate_all)(args.parent, args.root)
