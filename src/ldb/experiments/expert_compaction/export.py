"""Write an independent compressed checkpoint; never modify source weights."""

import argparse
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import make_shards

from .artifacts import sha
from .compact_model import validate_map


def export(source, destination, mappings, prompt_contract=None):
    source, destination = Path(source), Path(destination)
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "qwen2_moe" or config.get("expert_maps"):
        raise ValueError("Expected an original Qwen2 MoE checkpoint")
    if len(mappings) != config["num_hidden_layers"]:
        raise ValueError("Missing layer maps")
    retained = [validate_map(m, config["num_experts"]) for m in mappings]
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    weight_map, source_hashes, destination_hashes = {}, {}, {}
    weights = {}
    for filename in sorted(set(source_index["weight_map"].values())):
        source_hashes[filename] = sha(source / filename)
        incoming = mx.load(str(source / filename))
        if weights.keys() & incoming.keys():
            raise ValueError("Duplicate source tensor across shards")
        weights.update(incoming)
    del incoming
    if set(weights) != set(source_index["weight_map"]):
        raise ValueError("Source tensor index mismatch")
    original_bytes = sum(w.nbytes for w in weights.values())
    groups = {}
    for name in list(weights):
        if ".mlp.experts." in name:
            prefix, tail = name.split(".experts.")
            expert, suffix = tail.split(".", 1)
            groups.setdefault(f"{prefix}.switch_mlp.{suffix}", {})[int(expert)] = weights.pop(name)
        elif ".mlp.switch_mlp." in name:
            layer = int(name.split(".")[2])
            if weights[name].shape[0] != config["num_experts"]:
                raise ValueError(f"Unexpected expert axis: {name}")
            weights[name] = mx.take(weights[name], mx.array(retained[layer]), axis=0)
    for name, experts in groups.items():
        if name in weights or set(experts) != set(range(config["num_experts"])):
            raise ValueError(f"Incomplete or mixed expert layout: {name}")
        layer = int(name.split(".")[2])
        weights[name] = mx.stack([experts[e] for e in retained[layer]])
    groups.clear()
    compact_bytes = sum(w.nbytes for w in weights.values())
    shards = make_shards(weights)
    weights.clear()
    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        filename = (
            "model.safetensors"
            if len(shards) == 1
            else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        )
        mx.eval(shard)
        mx.save_safetensors(str(destination / filename), shard, metadata={"format": "mlx"})
        weight_map.update({name: filename for name in shard})
        destination_hashes[filename] = sha(destination / filename)
        del shard
        mx.clear_cache()
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    implementation = Path(__file__).with_name("compact_model.py")
    shutil.copy2(implementation, destination / "compact_model.py")
    shutil.copy2(Path(__file__).with_name("run_model.py"), destination / "run_model.py")
    if prompt_contract:
        contract = json.loads(Path(prompt_contract).read_text())
        prompt_source = Path(__file__).with_name("prompts.py")
        if sha(prompt_source) != contract["prompt_source_sha256"]:
            raise ValueError("Task prompt source differs from the frozen contract")
        shutil.copy2(prompt_source, destination / "prompts.py")
        shutil.copy2(prompt_contract, destination / "prompt-contract.json")
    (destination / "requirements.txt").write_text(
        "\n".join(
            f"{package}=={version(package)}"
            for package in ["mlx", "mlx-lm", "transformers", "numpy"]
        )
        + "\n"
    )
    config.update(expert_maps=mappings, model_file="compact_model.py")
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = {"metadata": {"total_size": compact_bytes}, "weight_map": weight_map}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": source_hashes,
        "weights_sha256": destination_hashes,
        "original_tensor_bytes": original_bytes,
        "compact_tensor_bytes": compact_bytes,
        "reduction": 1 - compact_bytes / original_bytes,
        "implementation_sha256": sha(implementation),
        "retained_per_layer": list(map(len, retained)),
        "metadata_sha256": {
            p.name: sha(p) for p in destination.iterdir() if p.suffix in {".py", ".json", ".txt"}
        },
        "semantics": "Original router slots/scores; physically shared donor weights; unchanged top-k",
    }
    (destination / "compaction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("mapping", type=Path, help="JSON list: one original-ID donor map per layer")
    parser.add_argument("--prompt-contract", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.source,
                args.destination,
                json.loads(args.mapping.read_text()),
                args.prompt_contract,
            ),
            indent=2,
        )
    )
