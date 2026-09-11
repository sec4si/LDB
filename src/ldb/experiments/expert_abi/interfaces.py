"""Low-capacity maps fitted once per direction, not once per held-out expert."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.optimize import linear_sum_assignment

from .common import aligned_layers, freeze, read, setup, verify, write
from .model import body, weights

PAIRS = [(0, 1), (0, 2), (1, 2)]
KINDS = ["diagonal", "rank8", "rank32"]


def mapping(x, params):
    if "scale" in params:
        return x * params["scale"] + params["bias"]
    result = (x @ params["left"]) @ params["right"] + params["bias"]
    return result + x if bool(params["identity"]) else result


def fit(x, y, kind, weight=None, identity=True):
    x, y = x.double(), y.double()
    w = torch.ones(len(x), dtype=torch.float64) if weight is None else weight.double().square()
    if (
        len(x) != len(y)
        or not torch.isfinite(x).all()
        or not torch.isfinite(y).all()
        or w.sum() <= 0
    ):
        raise ValueError("Invalid regression inputs")
    w = w / w.sum()
    mx, my = (w[:, None] * x).sum(0), (w[:, None] * y).sum(0)
    xc, yc = x - mx, y - my
    if kind == "diagonal":
        covariance = (w[:, None] * xc.square()).sum(0)
        ridge = max(float(covariance.mean()) * 0.001, 1e-12)
        prior = torch.ones_like(covariance) if identity else torch.zeros_like(covariance)
        scale = ((w[:, None] * xc * yc).sum(0) + ridge * prior) / (covariance + ridge)
        return dict(scale=scale.float(), bias=(my - mx * scale).float())
    rank = int(kind.removeprefix("rank"))
    target = yc - xc if identity else yc
    covariance = xc.T @ (w[:, None] * xc)
    ridge = max(float(covariance.trace()) / x.shape[1] * 0.001, 1e-12)
    coefficient = torch.linalg.solve(
        covariance + ridge * torch.eye(x.shape[1], dtype=x.dtype), xc.T @ (w[:, None] * target)
    )
    u, s, vh = torch.linalg.svd(coefficient, full_matrices=False)
    left, right = u[:, :rank] * s[:rank], vh[:rank]
    bias = my - mx @ left @ right - (mx if identity else 0)
    return dict(
        left=left.float(), right=right.float(), bias=bias.float(), identity=torch.tensor(identity)
    )


def save_maps(path, maps):
    save_file(
        {
            f"{name}.{key}": value.contiguous().clone()
            for name, params in maps.items()
            for key, value in params.items()
        },
        str(path),
    )


def load_maps(path):
    output = {}
    for key, value in load_file(str(path)).items():
        name, field = key.split(".")
        output.setdefault(name, {})[field] = value
    return output


def capture(root, model, split, layer):
    with np.load(root / "captures" / f"model-{model}" / split / f"layer-{layer}.npz") as data:
        return {k: torch.from_numpy(data[k].copy()) for k in ["x", "route", "y"]}


def cost(target, donors, route):
    """Normalized squared output error; zero donor has cost exactly one."""
    w = route.double().square()
    denominator = (w * target.double().square().sum(1)).sum()
    if denominator <= 1e-20:
        return [1e30] * len(donors)
    return [
        float((w * (z.double() - target.double()).square().sum(1)).sum() / denominator)
        for z in donors
    ]


def forward(x, payload, maps):
    return mapping(body(mapping(x, maps["H"]), payload), maps["T"])


def pair_cost(root, records, a, b, layers, split, maps_ab, maps_ba, parity):
    la, lb = layers
    ca, cb = capture(root, a, split, la), capture(root, b, split, lb)
    ea = list(range(parity, ca["route"].shape[1], 2))
    eb = list(range(parity, cb["route"].shape[1], 2))
    wa = [weights(records[a], la, e) for e in ea]
    wb = [weights(records[b], lb, e) for e in eb]
    with torch.no_grad():
        ya, yb = [body(ca["x"], p) for p in wa], [body(cb["x"], p) for p in wb]
        ba = [forward(ca["x"], p, maps_ab) for p in wb]
        ab = [forward(cb["x"], p, maps_ba) for p in wa]
        left = np.asarray([cost(y, ba, ca["route"][:, e]) for y, e in zip(ya, ea, strict=True)])
        right = np.asarray([cost(y, ab, cb["route"][:, e]) for y, e in zip(yb, eb, strict=True)]).T
    return np.maximum(left, right), ea, eb, ca, cb


def run_fit(root):
    records = setup(root)
    verify(root, "capture-freeze.json")
    config = [read(Path(r["path"]) / "config.json") for r in records]
    directory = root / "interfaces"
    directory.mkdir(exist_ok=False)
    training = []
    for a, b in PAIRS:
        layers = aligned_layers(config[a]["num_hidden_layers"], config[b]["num_hidden_layers"])
        train_layers = layers[::2]
        native = {}
        for source, target, column in [(a, b, 0), (b, a, 1)]:
            x = torch.cat(
                [capture(root, source, "fit", pair[column])["x"] for pair in train_layers]
            )
            y = torch.cat(
                [capture(root, target, "fit", pair[1 - column])["x"] for pair in train_layers]
            )
            native[(source, target)] = (x, y)
        for kind in KINDS:
            maps = {}
            for direction, (x, y) in native.items():
                d = x.shape[1]
                maps[direction] = dict(
                    H=fit(x, y, kind), T=dict(scale=torch.ones(d), bias=torch.zeros(d))
                )
            # Initial matching uses the reverse native-space linear map, without its absolute-state bias.
            for source, target in [(a, b), (b, a)]:
                provisional = {k: v.clone() for k, v in maps[(target, source)]["H"].items()}
                provisional["bias"].zero_()
                maps[(source, target)]["T"] = provisional
            fitrows = {(a, b): [], (b, a): []}
            matching = []
            for la, lb in train_layers:
                matrix, ea, eb, ca, cb = pair_cost(
                    root, records, a, b, (la, lb), "fit", maps[(a, b)], maps[(b, a)], 0
                )
                ii, jj = linear_sum_assignment(matrix)
                for i, j in zip(ii, jj, strict=True):
                    e, f = ea[i], eb[j]
                    matching.append(
                        dict(
                            a_layer=la, b_layer=lb, a_expert=e, b_expert=f, cost=float(matrix[i, j])
                        )
                    )
                    for source, target, data, sl, se, tl, te in [
                        (a, b, ca, la, e, lb, f),
                        (b, a, cb, lb, f, la, e),
                    ]:
                        x, w = data["x"], data["route"][:, se]
                        with torch.no_grad():
                            z = body(
                                mapping(x, maps[(source, target)]["H"]),
                                weights(records[target], tl, te),
                            )
                            y = body(x, weights(records[source], sl, se))
                        fitrows[(source, target)].append((x, z, y, w))
            for direction, rows in fitrows.items():
                x, z, y, w = [torch.cat([row[i] for row in rows]) for i in range(4)]
                maps[direction]["T"] = fit(z, y, kind, w)
                if kind == "rank32":
                    maps[direction]["predictor"] = fit(x, y, "rank64", w, identity=False)
                source, target = direction
                path = directory / f"{source}-from-{target}-{kind}.safetensors"
                save_maps(path, maps[direction])
                parameters = {
                    name: sum(v.numel() for k, v in params.items() if k != "identity")
                    for name, params in maps[direction].items()
                }
                with torch.no_grad():
                    error = float(
                        ((mapping(z, maps[direction]["T"]) - y).square().sum(1) * w.square()).sum()
                        / (y.square().sum(1) * w.square()).sum()
                    )
                training.append(
                    dict(
                        recipient=source,
                        donor=target,
                        kind=kind,
                        training_pairs=matching,
                        train_layer_pairs=train_layers,
                        heldout_layer_pairs=layers[1::2],
                        parameters=parameters,
                        fit_relative_error=error,
                        artifact=str(path.relative_to(root)),
                    )
                )
            print("Fitted shared interfaces", a, b, kind, flush=True)
    write(root / "interface-training.json", training)
    freeze(
        root,
        ["interface-training.json"] + [str(p.relative_to(root)) for p in directory.glob("*")],
        "interface-freeze.json",
    )


def select(root):
    records = setup(root)
    verify(root, "interface-freeze.json")
    config = [read(Path(r["path"]) / "config.json") for r in records]
    choices = []
    for a, b in PAIRS:
        layers = aligned_layers(config[a]["num_hidden_layers"], config[b]["num_hidden_layers"])[
            1::2
        ]
        ab = load_maps(root / "interfaces" / f"{a}-from-{b}-rank32.safetensors")
        ba = load_maps(root / "interfaces" / f"{b}-from-{a}-rank32.safetensors")
        candidates = []
        for la, lb in layers:
            matrix, ea, eb, ca, cb = pair_cost(
                root, records, a, b, (la, lb), "selection", ab, ba, 1
            )
            ac = (ca["route"] > 0).float().mean(0)
            bc = (cb["route"] > 0).float().mean(0)
            for i, e in enumerate(ea):
                for j, f in enumerate(eb):
                    if ac[e] < 0.02 or bc[f] < 0.02:
                        matrix[i, j] = 1e30
            ii, jj = linear_sum_assignment(matrix)
            for i, j in zip(ii, jj, strict=True):
                candidates.append(
                    dict(
                        a_layer=la,
                        b_layer=lb,
                        a_expert=ea[i],
                        b_expert=eb[j],
                        cost=float(matrix[i, j]),
                        a_coverage=float(ac[ea[i]]),
                        b_coverage=float(bc[eb[j]]),
                    )
                )
        ranked = sorted(candidates, key=lambda r: r["cost"])
        selected = []
        for candidate in ranked:
            if candidate["cost"] >= 1e29:
                continue
            if sum(r["a_layer"] == candidate["a_layer"] for r in selected) >= 2:
                continue
            selected.append(candidate)
            if len(selected) == 4:
                break
        if len(selected) < 4 or len({r["a_layer"] for r in selected}) < 2:
            raise ValueError("Insufficient eligible held-out expert pairs")
        choices.append(
            dict(
                a=a,
                b=b,
                candidates=candidates,
                selected=selected,
                all_same_slot=all(
                    p["a_layer"] == p["b_layer"] and p["a_expert"] == p["b_expert"]
                    for p in selected
                ),
            )
        )
        print("Selected pairs", a, b, selected, flush=True)
    write(root / "choices.json", choices)
    freeze(
        root,
        ["choices.json", "interface-freeze.json", "input-freeze.json"],
        "selection-freeze.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "select"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    (run_fit if args.stage == "fit" else select)(args.root)
