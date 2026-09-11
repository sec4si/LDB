"""Optimize independent linear formats and a tied-GU control on the same losses."""

import argparse
import json
import math
from pathlib import Path

from .qwen_mixed_loss_plan import choose

SHAPES = {"gate": (1408, 2048), "up": (1408, 2048), "down": (2048, 1408)}


def size(part, fmt):
    n, k = SHAPES[part]
    return n * k // 2 + 4 * n if fmt == "w4a16" else n * k + 2 * n


def optimize(values, reference):
    layers = len(reference)
    base_loss = sum(
        values[li, e, p, "w4a16"] for li in range(layers) for e in range(60) for p in SHAPES
    )
    gains = {
        (li, e, p): values[li, e, p, "w4a16"] - values[li, e, p, "w8a8"]
        for li in range(layers)
        for e in range(60)
        for p in SHAPES
    }
    target = sum(size(p, c[p]) for layer in reference for c in layer for p in SHAPES)
    base_bytes = layers * 60 * sum(size(p, "w4a16") for p in SHAPES)
    costs = {p: size(p, "w8a8") - size(p, "w4a16") for p in SHAPES}
    results = {}
    for tied in (False, True):
        groups = {"gu": [], "down": []}
        for li in range(layers):
            for e in range(60):
                for parts in (
                    [("gate", "up"), ("down",)] if tied else [("gate",), ("up",), ("down",)]
                ):
                    group = "down" if parts == ("down",) else "gu"
                    groups[group].append(
                        dict(
                            layer=li,
                            expert=e,
                            parts=parts,
                            gain=sum(gains[li, e, p] for p in parts),
                        )
                    )
        selected, gain = choose(
            groups,
            target - base_bytes,
            dict(gu=costs["gate"] * (2 if tied else 1), down=costs["down"]),
        )
        plan = [[{p: "w4a16" for p in SHAPES} for _ in range(60)] for _ in range(layers)]
        for r in selected:
            for p in r["parts"]:
                plan[r["layer"]][r["expert"]][p] = "w8a8"
        actual = sum(size(p, c[p]) for layer in plan for c in layer for p in SHAPES)
        assert actual <= target
        results["tied" if tied else "independent"] = dict(
            plan=plan,
            predicted_additive_loss=base_loss - gain,
            routed_tensor_bytes=actual,
            reference_routed_tensor_bytes=target,
        )
    assert (
        results["independent"]["predicted_additive_loss"]
        <= results["tied"]["predicted_additive_loss"] + 1e-9
    )
    reference_loss = base_loss - sum(
        gains[li, e, p]
        for li, layer in enumerate(reference)
        for e, c in enumerate(layer)
        for p in SHAPES
        if c[p] == "w8a8"
    )
    for result in results.values():
        result["reference_predicted_additive_loss"] = reference_loss
    return results


def run(profile, reference, output):
    complete = json.loads((profile / "complete.json").read_text())
    assert complete["candidates_per_layer"] == 360
    ref = json.loads(reference.read_text())["plan"]
    ref = [
        [dict(gate=c["gu"], up=c["gu"], down=c["down"]) if "gu" in c else c for c in layer]
        for layer in ref
    ]
    assert len(ref) == complete["layers"] and all(len(layer) == 60 for layer in ref)
    assert all(
        set(c) == set(SHAPES) and all(v in ("w4a16", "w8a8") for v in c.values())
        for layer in ref
        for c in layer
    )
    values = {}
    for li in range(len(ref)):
        data = json.loads((profile / f"layer-{li}/result.json").read_text())
        assert data["windows"] == complete["windows"] and len(data["rows"]) == 360
        for row in data["rows"]:
            key = (li, row["expert"], row["part"], row["format"])
            assert key not in values
            assert math.isfinite(row["loss"]) and row["loss"] >= 0
            values[key] = row["loss"]
    results = optimize(values, ref)
    output.mkdir(parents=True, exist_ok=False)
    for name, result in results.items():
        result.update(
            calibration_windows=complete["windows"],
            scope="Exact additive-loss optimum for selected W4A16/W8A8 search space; no runtime objective, GPTQ or model quality claim",
        )
        (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return {
        name: {k: v for k, v in result.items() if k != "plan"} for name, result in results.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("profile", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
