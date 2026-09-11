"""Backend-free IR validation, serialization and structural comparison."""

import hashlib
import json
import math
import re
from pathlib import Path

VERSION = "moe.ir/v1"
BOUNDARY = {
    "input": "post_moe_norm",
    "output": "unweighted_expert_update",
    "routing": "host",
    "residual": "host",
    "compute_dtype": "float32",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) is int and value > 0


def validate_graph(graph):
    require(set(graph) == {"input_width", "parameters", "nodes", "output"}, "Unknown graph fields")
    require(positive(graph["input_width"]), "Invalid graph input width")
    widths = {"x": graph["input_width"]}
    used = set()
    for node in graph["nodes"]:
        name, op, inputs = node["id"], node["op"], node["inputs"]
        require(
            set(node)
            == ({"id", "op", "inputs", "weight"} if op == "linear" else {"id", "op", "inputs"}),
            "Unknown operation fields",
        )
        require(
            isinstance(name, str) and name and name not in widths, "Duplicate or invalid SSA value"
        )
        require(
            isinstance(inputs, list) and all(i in widths for i in inputs),
            "Undefined or forward SSA reference",
        )
        if op == "linear":
            require(len(inputs) == 1, "linear requires one input")
            parameter = node["weight"]
            require(parameter in graph["parameters"], "Undefined weight binding")
            shape = graph["parameters"][parameter]
            require(len(shape) == 2 and all(positive(v) for v in shape), "Invalid parameter shape")
            require(shape[1] == widths[inputs[0]], "linear input width mismatch")
            widths[name] = shape[0]
            used.add(parameter)
        elif op == "silu":
            require(len(inputs) == 1, "silu requires one input")
            widths[name] = widths[inputs[0]]
        elif op == "multiply":
            require(
                len(inputs) == 2 and widths[inputs[0]] == widths[inputs[1]],
                "multiply shape mismatch",
            )
            widths[name] = widths[inputs[0]]
        else:
            raise ValueError(f"Unsupported IR operation: {op}")
    require(graph["output"] in widths and graph["output"] != "x", "Missing graph output")
    require(widths[graph["output"]] == graph["input_width"], "Expert must preserve host width")
    require(used == set(graph["parameters"]), "Unused parameter declaration")


def validate(ir):
    """Reject malformed graphs before resolving or reading model payloads."""
    try:
        require(ir["schema"] == VERSION, "Unsupported IR version")
        require(ir["boundary"] == BOUNDARY, "Unsupported host boundary or compute dtype")
        require(
            isinstance(ir["model"]["name"], str) and ir["model"]["name"], "Missing model identity"
        )
        for source in ir["sources"].values():
            require(isinstance(source["uri"], str) and source["uri"], "Missing source URI")
            require(
                re.fullmatch("[0-9a-f]{64}", source["sha256"]) is not None, "Invalid source hash"
            )
        for graph in ir["graphs"].values():
            validate_graph(graph)
        require(bool(ir["layers"]), "Empty layer inventory")
        layers = set()
        for layer in ir["layers"]:
            lid = layer["id"]
            require(
                type(lid) is int and lid >= 0 and lid not in layers, "Duplicate or invalid layer ID"
            )
            layers.add(lid)
            require(layer["graph"] in ir["graphs"], "Unknown graph")
            graph = ir["graphs"][layer["graph"]]
            routing = layer["routing"]
            require(
                positive(routing["experts"])
                and positive(routing["top_k"])
                and routing["top_k"] <= routing["experts"],
                "Invalid routing count",
            )
            require(
                routing["shared_experts"] == 0 and routing["weighting"] == "host_defined",
                "Unsupported routing contract",
            )
            ids = set()
            for expert in layer["experts"]:
                eid = expert["id"]
                require(
                    type(eid) is int and 0 <= eid < routing["experts"] and eid not in ids,
                    "Duplicate or invalid expert ID",
                )
                ids.add(eid)
                require(
                    set(expert["bindings"]) == set(graph["parameters"]),
                    "Incomplete expert bindings",
                )
                for ref in expert["bindings"].values():
                    require(ref["source"] in ir["sources"], "Unknown tensor source")
                    require(isinstance(ref["key"], str) and ref["key"], "Invalid tensor key")
                    require(
                        ref["storage_dtype"] in ["F32", "F16", "BF16"],
                        "Unsupported tensor storage dtype",
                    )
            require(ids == set(range(routing["experts"])), "Incomplete layer expert inventory")
            if "observations" in layer:
                evidence = layer["observations"]
                require(
                    evidence["scope"] == "corpus_conditioned"
                    and evidence["source"] in ir["sources"],
                    "Labels require evidence provenance",
                )
                require(isinstance(evidence["cluster"], str), "Invalid cluster label")
                require(
                    all(
                        type(v) in [int, float] and math.isfinite(v)
                        for v in evidence["metrics"].values()
                    ),
                    "Nonfinite observed metric",
                )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed IR: {error}") from error
    return ir


def load(path):
    return validate(json.loads(Path(path).read_text()))


def save(path, ir):
    validate(ir)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ir, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def layer_at(ir, layer):
    return next(row for row in ir["layers"] if row["id"] == layer)


def compatibility(left, left_layer, right, right_layer):
    """A static check, deliberately unable to certify functional equivalence."""
    validate(left)
    validate(right)
    a, b = layer_at(left, left_layer), layer_at(right, right_layer)
    checks = {
        "boundary": left["boundary"] == right["boundary"],
        "graph": left["graphs"][a["graph"]] == right["graphs"][b["graph"]],
        "routing": a["routing"] == b["routing"],
    }
    return {
        "expert_structure_compatible": checks["boundary"] and checks["graph"],
        "whole_layer_structure_compatible": all(checks.values()),
        "checks": checks,
        "functional_compatibility": "unverified",
        "reason": "Shape/operator equality and matching profile labels do not establish interchangeable learned functions.",
    }


def swiglu(width, intermediate):
    return {
        "input_width": width,
        "parameters": {
            "gate": [intermediate, width],
            "up": [intermediate, width],
            "down": [width, intermediate],
        },
        "nodes": [
            {"id": "g", "op": "linear", "inputs": ["x"], "weight": "gate"},
            {"id": "u", "op": "linear", "inputs": ["x"], "weight": "up"},
            {"id": "a", "op": "silu", "inputs": ["g"]},
            {"id": "m", "op": "multiply", "inputs": ["a", "u"]},
            {"id": "y", "op": "linear", "inputs": ["m"], "weight": "down"},
        ],
        "output": "y",
    }
