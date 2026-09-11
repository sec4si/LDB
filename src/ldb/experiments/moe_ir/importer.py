"""Import the audited native Mixtral checkpoint layout into portable IR."""

import json
from pathlib import Path

from safetensors import safe_open

from .core import BOUNDARY, VERSION, digest, require, swiglu, validate


def from_study(root, model):
    root = Path(root).resolve()
    frozen = json.loads((root / "capture-freeze.json").read_text())
    require(
        digest(root / "layer-contracts.json") == frozen["layer-contracts.json"],
        "Frozen layer contracts changed",
    )
    records = json.loads((root / "models.json").read_text())
    record = records[model]
    checkpoint = Path(record["path"]) / "model.safetensors"
    require(digest(checkpoint) == record["weight_sha256"], "Checkpoint identity changed")
    require(
        digest(Path(record["path"]) / "config.json") == record["config_sha256"],
        "Config identity changed",
    )
    contracts = json.loads((root / "layer-contracts.json").read_text())
    label_path = root / "layer-labels.json"
    labels = json.loads(label_path.read_text())
    ir = {
        "schema": VERSION,
        "model": {k: record[k] for k in ["name", "revision", "config_sha256"]},
        "boundary": dict(BOUNDARY),
        "sources": {
            "weights": {"uri": str(checkpoint), "sha256": record["weight_sha256"]},
            "observations": {"uri": str(label_path), "sha256": digest(label_path)},
            "contracts": {
                "uri": str(root / "layer-contracts.json"),
                "sha256": digest(root / "layer-contracts.json"),
            },
        },
        "graphs": {},
        "layers": [],
        "coverage": "MoE expert subgraphs only; attention, routing and residual execution remain in the host",
    }
    with safe_open(checkpoint, framework="pt") as tensors:
        for row in contracts:
            if row["model"] != model:
                continue
            require(
                row["operator"] == "down(silu(gate(x))*up(x))" and row["shared_experts"] == 0,
                "Unsupported importer architecture",
            )
            require(
                not any(row[k] for k in ["gate_bias", "up_bias", "down_bias"]),
                "Biased projections unsupported",
            )
            graph_id = f"swiglu-{row['hidden_size']}-{row['intermediate_size']}"
            graph = swiglu(row["hidden_size"], row["intermediate_size"])
            ir["graphs"][graph_id] = graph
            layer = {
                "id": row["layer"],
                "graph": graph_id,
                "routing": {
                    "experts": row["experts"],
                    "top_k": row["top_k"],
                    "shared_experts": 0,
                    "weighting": "host_defined",
                },
                "experts": [],
            }
            for expert in range(row["experts"]):
                bindings = {}
                for role, native in [("gate", "w1"), ("up", "w3"), ("down", "w2")]:
                    key = f"model.layers.{row['layer']}.block_sparse_moe.experts.{expert}.{native}.weight"
                    tensor = tensors.get_slice(key)
                    require(
                        tensor.get_shape() == graph["parameters"][role],
                        f"Tensor shape mismatch: {key}",
                    )
                    bindings[role] = {
                        "source": "weights",
                        "key": key,
                        "storage_dtype": tensor.get_dtype(),
                    }
                layer["experts"].append({"id": expert, "bindings": bindings})
            observed = next(
                x for x in labels["layers"] if x["model"] == model and x["layer"] == row["layer"]
            )
            layer["observations"] = {
                "scope": "corpus_conditioned",
                "source": "observations",
                "cluster": observed["heldout_cluster"],
                "metrics": {k: observed[k] for k in labels["features"]},
                "semantic_role": "unassigned",
            }
            ir["layers"].append(layer)
    return validate(ir)
