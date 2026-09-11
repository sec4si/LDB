"""Finite candidate plans. Surrogate DP never substitutes for full-model gates."""

import math
from pathlib import Path

from safetensors import safe_open

from ldb.experiments.expert_compaction.selection import allocate_layers
from ldb.experiments.moe_ir import compatibility
from ldb.experiments.moe_ir import load as load_ir

from .common import read, sha, validate_spec

BYTES = {"F32": 4, "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "U8": 1, "I8": 1, "I64": 8}


def inventory(host):
    root = Path(host["path"])
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError("No checkpoint weights")
    total, expert_bytes = 0, {}
    hashes = {}
    for path in files:
        hashes[path.name] = sha(path)
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                t = f.get_slice(key)
                size = math.prod(t.get_shape()) * BYTES[t.get_dtype()]
                total += size
                if ".mlp.experts." in key:
                    parts = key.split(".")
                    slot = (int(parts[2]), int(parts[5]))
                    expert_bytes[slot] = expert_bytes.get(slot, 0) + size
    expected = host.get("weight_sha256")
    if expected and hashes != expected:
        raise ValueError("Catalog checkpoint hash mismatch")
    config_hash = sha(root / "config.json")
    if host.get("config_sha256") and config_hash != host["config_sha256"]:
        raise ValueError("Catalog config hash mismatch")
    for name, expected in host.get("tokenizer_sha256", {}).items():
        if sha(Path(host["tokenizer"]) / name) != expected:
            raise ValueError("Catalog tokenizer identity mismatch")
    config = read(root / "config.json")
    precision = 2 if host["backend"] == "mlx" else 4
    return {
        "tensor_bytes": total,
        "weight_sha256": hashes,
        "config_sha256": config_hash,
        "expert_bytes": expert_bytes,
        "kv_bytes_per_token": 2
        * config["num_hidden_layers"]
        * config.get("num_key_value_heads", config["num_attention_heads"])
        * (config["hidden_size"] // config["num_attention_heads"])
        * precision,
    }


def compact_candidates(host, inv, spec):
    previous = Path(host["calibration"])
    frozen = read(previous / "candidate-freeze.json")
    if (
        inv["weight_sha256"] != frozen["source_weights"]
        or inv["config_sha256"] != frozen["source_config_sha256"]
    ):
        raise ValueError("Calibration belongs to another checkpoint")
    config = read(Path(host["path"]) / "config.json")
    counts = [0, 4, 8, 12, 16]
    sizes = set(inv["expert_bytes"].values())
    if (
        len(sizes) != 1
        or len(inv["expert_bytes"]) != config["num_hidden_layers"] * config["num_experts"]
    ):
        raise ValueError(
            "v1 compaction planner requires complete equal-size original expert tensors"
        )
    size = sizes.pop()
    # Leave 1 MiB for safetensors headers; actual export file size is gated later.
    needed = max(0, math.ceil((inv["tensor_bytes"] + 2**20 - spec["weight_bytes"]) / size))
    minimum = 4 * math.ceil(needed / 4)
    tables, costs = [], []
    for layer in range(config["num_hidden_layers"]):
        rows = {}
        for count in counts[1:]:
            name = f"fingerprint-layer{layer}-remove{count}.json"
            path = previous / "layer-search" / name
            if sha(path) != frozen["layer_search_artifacts"][name]:
                raise ValueError("Calibration layer search changed")
            rows[count] = read(path)
        tables.append(rows)
        costs.append([0.0, *[rows[c]["delta_nll"] for c in counts[1:]]])
    output = []
    for remove in range(minimum, minimum + 4 * spec["candidate_limit"], 4):
        if remove == 0 or remove > config["num_hidden_layers"] * 16:
            continue
        cost, allocation = allocate_layers(costs, counts, remove)
        maps = [
            list(range(config["num_experts"])) if n == 0 else tables[layer_row][n]["mapping"]
            for layer_row, n in enumerate(allocation)
        ]
        output.append(
            {
                "id": f"{host['id']}-compact-{remove}",
                "host": host["id"],
                "operation": "compact",
                "maps": maps,
                "removed": remove,
                "estimated_weight_bytes": inv["tensor_bytes"] - remove * size + 2**20,
                "surrogate_delta_nll": cost,
                "allocation": allocation,
                "calibration_freeze_sha256": sha(previous / "candidate-freeze.json"),
            }
        )
    return output


def plan(catalog, spec):
    validate_spec(spec)
    hosts = {h["id"]: h for h in catalog["hosts"]}
    if len(hosts) != len(catalog["hosts"]):
        raise ValueError("Duplicate host ID")
    inventories = {key: inventory(host) for key, host in hosts.items()}
    candidates = []
    for key, host in hosts.items():
        inv = inventories[key]
        candidates.append(
            {
                "id": f"{key}-original",
                "host": key,
                "operation": "original",
                "estimated_weight_bytes": sum(
                    (Path(host["path"]) / p).stat().st_size for p in inv["weight_sha256"]
                ),
            }
        )
        if host.get("calibration"):
            candidates.extend(compact_candidates(host, inv, spec))
    for swap in catalog.get("swaps", []):
        host, donor = hosts[swap["host"]], hosts[swap["donor"]]
        if host["backend"] != donor["backend"] or host["backend"] != "torch":
            raise ValueError("v1 swap requires two audited Torch hosts")
        a, b = load_ir(host["ir"]), load_ir(donor["ir"])
        for item in [a, b]:
            model_host = host if item is a else donor
            if (
                item["sources"]["weights"]["sha256"]
                != inventories[model_host["id"]]["weight_sha256"]["model.safetensors"]
            ):
                raise ValueError("IR identity differs from catalog weights")
        seen = set()
        for slot in swap["slots"]:
            target = (slot["layer"], slot["expert"])
            if target in seen:
                raise ValueError("Duplicate swap slot")
            seen.add(target)
            if not compatibility(a, slot["layer"], b, slot["donor_layer"])[
                "expert_structure_compatible"
            ]:
                raise ValueError("Incompatible IR expert structures")
        candidates.append(
            {
                "id": swap["id"],
                "host": host["id"],
                "donor": donor["id"],
                "operation": "swap",
                "slots": swap["slots"],
                "estimated_weight_bytes": inventories[host["id"]]["tensor_bytes"] + 2**20,
            }
        )
    for c in candidates:
        inv = inventories[c["host"]]
        c["kv_reserve_bytes"] = (
            inv["kv_bytes_per_token"] * spec["context_tokens"] * spec["concurrency"]
        )
        c["preflight_pass"] = c["estimated_weight_bytes"] <= spec["weight_bytes"] and c[
            "estimated_weight_bytes"
        ] + c["kv_reserve_bytes"] + 128 * 2**20 <= spec["memory_bytes"] * (
            1 - spec["memory_headroom"]
        )
    serial = {
        k: {name: value for name, value in row.items() if name != "expert_bytes"}
        for k, row in inventories.items()
    }
    return {
        "hosts": serial,
        "candidates": candidates,
        "scope": "Finite catalog and frozen transforms; estimates do not certify runtime fit or capability",
    }
