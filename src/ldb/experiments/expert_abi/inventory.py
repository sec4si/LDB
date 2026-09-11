"""Additional on-disk structure evidence, explicitly separate from full-model runs."""

import argparse
import re
from pathlib import Path

from safetensors import safe_open

from .common import read, sha, write


def inspect(path, kind):
    config = read(path / "config.json")
    files = sorted(path.glob("model*.safetensors"))
    tensors, hashes = {}, {}
    for filename in files:
        with safe_open(filename, framework="pt") as stream:
            for name in stream.keys():
                sliced = stream.get_slice(name)
                tensors[name] = dict(shape=sliced.get_shape(), dtype=sliced.get_dtype())
        hashes[filename.name] = sha(filename)
    layers = {}
    if kind == "physics":
        pattern = r"blocks\.(\d+)\.ffn\.(gate|up|down)$"
    else:
        pattern = (
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
        )
    for name, tensor in tensors.items():
        match = re.fullmatch(pattern, name)
        if match:
            layers.setdefault(int(match[1]), {})[name] = tensor
    expected = config.get("num_hidden_layers", config.get("n_layers"))
    result = dict(
        path=str(path),
        kind=kind,
        configured_layers=expected,
        model_type=config.get("model_type", config.get("_model_class")),
        hidden_size=config.get("hidden_size", config.get("d_model")),
        expert_width=config.get("moe_intermediate_size", config.get("d_ff")),
        configured_experts=config.get("num_experts", config.get("n_routed_experts")),
        top_k=config.get("num_experts_per_tok", config.get("top_k")),
        activation=config.get("hidden_act", "not_declared_in_config"),
        shared_expert_width=config.get("shared_expert_intermediate_size"),
        shared_expert_count=config.get("n_shared_experts"),
        dense_prefix_layers=config.get("moe_first_layer", 0),
        norm_topk_prob=config.get("norm_topk_prob"),
        quantization=config.get("quantization"),
        observed_expert_layers=sorted(layers),
        observed_expert_layer_count=len(layers),
        full_model_executed=False,
        coverage="partial checkpoint"
        if len(layers) < expected - config.get("moe_first_layer", 0)
        else "all configured MoE layers present",
        files=hashes,
        config_sha256=sha(path / "config.json"),
        layers=[],
    )
    for layer, values in sorted(layers.items()):
        result["layers"].append(
            dict(
                layer=layer,
                projection_tensor_count=len(values),
                representative_tensors=dict(list(sorted(values.items()))[:3]),
            )
        )
    return result


def run(root, base):
    data = [
        inspect(base / name, kind)
        for name, kind in [
            ("qwen1.5-moe-chat-mlx-4bit", "qwen2_moe_quantized"),
            ("qwen3-prefix", "qwen3_moe_prefix"),
            ("physics-moe", "physics"),
        ]
    ]
    write(
        root / "additional-structure.json",
        dict(
            models=data,
            source_sha256=sha(Path(__file__)),
            scope="Local tensor/config inventory only; quantized stored shapes are not logical dense shapes; no cross-architecture execution or interchange claim",
        ),
    )
    print(
        [(r["model_type"], r["observed_expert_layer_count"], r["coverage"]) for r in data],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.base)
