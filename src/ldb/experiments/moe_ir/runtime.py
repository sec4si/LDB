"""CPU FP32 reference lowering of validated expert graphs."""

from copy import deepcopy
from pathlib import Path

from .core import digest, layer_at, require, validate


class Runtime:
    """Resolve and verify weight assets once, then bind individual expert graphs."""

    def __init__(self, ir, base_dir=".", source_overrides=None):
        import torch
        from safetensors import safe_open

        self.ir = ir = deepcopy(validate(ir))
        self.weights = {}
        overrides = source_overrides or {}
        used = {
            ref["source"]
            for layer in ir["layers"]
            for expert in layer["experts"]
            for ref in expert["bindings"].values()
        }
        require(set(overrides) <= used, "Unknown weight source override")
        for source in used:
            path = Path(overrides.get(source, ir["sources"][source]["uri"]))
            path = path if path.is_absolute() else Path(base_dir) / path
            require(
                digest(path) == ir["sources"][source]["sha256"],
                f"Weight integrity mismatch: {source}",
            )
            with safe_open(path, framework="pt") as tensors:
                for layer in ir["layers"]:
                    graph = ir["graphs"][layer["graph"]]
                    for expert in layer["experts"]:
                        for role, ref in expert["bindings"].items():
                            if ref["source"] != source:
                                continue
                            value = tensors.get_slice(ref["key"])
                            require(
                                value.get_shape() == graph["parameters"][role],
                                "Stored tensor shape mismatch",
                            )
                            require(
                                value.get_dtype() == ref["storage_dtype"], "Stored dtype mismatch"
                            )
                            self.weights[layer["id"], expert["id"], role] = (
                                tensors.get_tensor(ref["key"]).to(torch.float32).contiguous()
                            )

    def expert(self, layer, expert):
        import torch
        import torch.nn.functional as functional

        graph = self.ir["graphs"][layer_at(self.ir, layer)["graph"]]
        parameters = {role: self.weights[layer, expert, role] for role in graph["parameters"]}

        def execute(x):
            require(
                x.device.type == "cpu"
                and x.dtype == torch.float32
                and x.ndim >= 1
                and x.shape[-1] == graph["input_width"],
                "IR input violates CPU FP32 width contract",
            )
            values = {"x": x}
            for node in graph["nodes"]:
                operands = [values[key] for key in node["inputs"]]
                if node["op"] == "linear":
                    result = functional.linear(operands[0], parameters[node["weight"]])
                elif node["op"] == "silu":
                    result = functional.silu(operands[0])
                else:
                    result = operands[0] * operands[1]
                values[node["id"]] = result
            return values[graph["output"]]

        return execute
