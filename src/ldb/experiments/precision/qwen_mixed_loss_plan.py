"""Exactly minimize additive layer-output losses under a routed tensor byte budget."""

import argparse
import itertools
import json
import math
from pathlib import Path

SHAPES = {"gu": (2816, 2048), "down": (2048, 1408)}


def size(part, fmt):
    n, k = SHAPES[part]
    return n * k // 2 + 4 * n if fmt == "w4a16" else n * k + 2 * n


def choose(groups, budget, costs):
    ranked = {
        p: sorted((r for r in groups[p] if r["gain"] > 0), key=lambda r: -r["gain"]) for p in costs
    }
    prefixes = {p: [0.0] for p in costs}
    for p, rows in ranked.items():
        for row in rows:
            prefixes[p].append(prefixes[p][-1] + row["gain"])
    best = (0.0, 0, 0)
    for gu in range(min(len(ranked["gu"]), budget // costs["gu"]) + 1):
        down = min(len(ranked["down"]), (budget - gu * costs["gu"]) // costs["down"])
        gain = prefixes["gu"][gu] + prefixes["down"][down]
        if gain > best[0]:
            best = (gain, gu, down)
    gain, gu, down = best
    return ranked["gu"][:gu] + ranked["down"][:down], gain


def run(profile, reference, output):
    complete = json.loads((profile / "complete.json").read_text())
    ref = json.loads(reference.read_text())["plan"]
    assert complete["layers"] == len(ref) == 24
    groups = {p: [] for p in SHAPES}
    base_loss = 0.0
    for li in range(24):
        data = json.loads((profile / f"layer-{li}/result.json").read_text())
        assert data["windows"] == complete["windows"]
        values = {(r["expert"], r["part"], r["format"]): r["loss"] for r in data["rows"]}
        assert len(values) == len(data["rows"]) == 240
        assert all(math.isfinite(v) and v >= 0 for v in values.values())
        for e, p in itertools.product(range(60), SHAPES):
            low = values[e, p, "w4a16"]
            high = values[e, p, "w8a8"]
            base_loss += low
            groups[p].append(dict(layer=li, expert=e, part=p, gain=low - high))
    assert all(len(layer) == 60 for layer in ref)
    assert all(
        set(c) == set(SHAPES) and all(v in ("w4a16", "w8a8") for v in c.values())
        for layer in ref
        for c in layer
    )
    target = sum(size(p, choice[p]) for layer in ref for choice in layer for p in SHAPES)
    base_bytes = 24 * 60 * sum(size(p, "w4a16") for p in SHAPES)
    costs = {p: size(p, "w8a8") - size(p, "w4a16") for p in SHAPES}
    selected, gain = choose(groups, target - base_bytes, costs)
    plan = [[dict(gu="w4a16", down="w4a16") for _ in range(60)] for _ in range(24)]
    for row in selected:
        plan[row["layer"]][row["expert"]][row["part"]] = "w8a8"
    actual = sum(size(p, c[p]) for layer in plan for c in layer for p in SHAPES)
    assert actual <= target
    reference_gain = sum(
        r["gain"]
        for rows in groups.values()
        for r in rows
        if ref[r["layer"]][r["expert"]][r["part"]] == "w8a8"
    )
    assert gain >= reference_gain - 1e-10
    result = dict(
        plan=plan,
        routed_tensor_bytes=actual,
        reference_routed_tensor_bytes=target,
        predicted_additive_loss=base_loss - gain,
        all_w4_additive_loss=base_loss,
        reference_predicted_additive_loss=base_loss - reference_gain,
        upgrades={p: sum(r["part"] == p for r in selected) for p in SHAPES},
        calibration_windows=complete["windows"],
        scope="Exact optimum of measured additive loss for two fixed block sizes and formats; GU tied; not full-paper search or measured model quality",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "plan"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("profile", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
