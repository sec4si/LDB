"""One fresh process per build or measurement; result JSON is the protocol."""

import argparse
import math
import shutil
import time
from pathlib import Path

from .common import read, sha, write
from .prompts import messages


def build(root, candidate, hosts):
    host = hosts[candidate["host"]]
    source = Path(host["path"])
    destination = root / "artifacts" / candidate["id"]
    if candidate["operation"] == "original":
        return source
    if candidate["operation"] == "compact":
        from ldb.experiments.expert_compaction.export import export

        export(source, destination, candidate["maps"])
    else:
        from safetensors.torch import load_file, save_file

        values = load_file(str(source / "model.safetensors"))
        donor = load_file(str(Path(hosts[candidate["donor"]]["path"]) / "model.safetensors"))
        for slot in candidate["slots"]:
            for name in ["w1", "w2", "w3"]:
                a = f"model.layers.{slot['layer']}.block_sparse_moe.experts.{slot['expert']}.{name}.weight"
                b = f"model.layers.{slot['donor_layer']}.block_sparse_moe.experts.{slot['donor_expert']}.{name}.weight"
                if values[a].shape != donor[b].shape:
                    raise ValueError("Donor tensor shape mismatch")
                values[a] = donor[b].to(values[a].dtype).contiguous()
        destination.mkdir(parents=True, exist_ok=False)
        save_file(values, str(destination / "model.safetensors"), metadata={"format": "pt"})
        for f in source.iterdir():
            if f.suffix in [".json", ".txt", ".model"] and not f.name.endswith(
                ".safetensors.index.json"
            ):
                shutil.copy2(f, destination / f.name)
    write(
        destination / "build-provenance.json",
        {
            "candidate": candidate,
            "host": host["id"],
            "weight_sha256": {p.name: sha(p) for p in destination.glob("*.safetensors")},
        },
    )
    return destination


class TorchModel:
    def __init__(self, path, tokenizer_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            experts_implementation="eager",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or path, local_files_only=True
        )

    def nll(self, ids):
        import torch

        x = torch.tensor([ids])
        logits = self.model(x[:, :-1], use_cache=False).logits.double()
        loss = torch.nn.functional.cross_entropy(logits[0], x[0, 1:], reduction="sum")
        return float(loss)

    def generate_ids(self, ids, count, stop=True):
        import torch

        x, cache, output, stamps = torch.tensor([ids]), None, [], []
        for _ in range(count):
            result = self.model(x, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            token = int(result.logits[0, -1].argmax())
            stamps.append(time.perf_counter())
            output.append(token)
            if stop and token == self.tokenizer.eos_token_id:
                break
            x = torch.tensor([[token]])
        return output, stamps


class MLXModel:
    def __init__(self, path, tokenizer_path=None):
        import mlx.core as mx
        from mlx_lm import load

        mx.set_cache_limit(256 * 2**20)
        self.model, self.tokenizer = load(str(path))
        self.model.eval()
        mx.eval(self.model.parameters())
        mx.clear_cache()

    def nll(self, ids):
        import mlx.core as mx

        x = mx.array([ids])
        logits = self.model(x[:, :-1]).astype(mx.float32)
        logs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        loss = -mx.take_along_axis(logs, x[:, 1:, None], axis=-1).sum()
        return float(loss.item())

    def generate_ids(self, ids, count, stop=True):
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        output, stamps = [], []
        for token, _ in generate_step(mx.array(ids), self.model, max_tokens=count):
            output.append(int(token))
            stamps.append(time.perf_counter())
            if stop and token == self.tokenizer.eos_token_id:
                break
        mx.synchronize()
        return output, stamps


def evaluate(path, backend, data, spec, tokenizer_path=None):
    started = time.perf_counter()
    model = (MLXModel if backend == "mlx" else TorchModel)(path, tokenizer_path)
    tok = model.tokenizer
    documents, tasks = [], []
    for text in data["texts"]:
        ids = tok.encode(text, add_special_tokens=False)
        if not 2 <= len(ids) <= spec["context_tokens"]:
            raise ValueError("Text exceeds declared workload or has no next-token targets")
        nll = model.nll(ids)
        documents.append(
            {
                "nll": nll / (len(ids) - 1),
                "tokens": len(ids) - 1,
                "bits_per_byte": nll / math.log(2) / len(text.encode("utf-8")),
            }
        )
    for task in data["tasks"]:
        chat = messages(task)
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\nassistant:"
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) + spec["max_new_tokens"] > spec["context_tokens"]:
            raise ValueError("Task does not fit declared context; no silent truncation")
        output, _ = model.generate_ids(ids, spec["max_new_tokens"])
        text = tok.decode(output, skip_special_tokens=True).strip()
        tasks.append(
            {
                "id": task["id"],
                "kind": task["kind"],
                "answer": text,
                "correct": text == task["answer"],
                "generated_tokens": len(output),
            }
        )
    # Explicit capacity workload reaches the configured context length including KV.
    pattern = tok.encode("A library stores books for people to read. ", add_special_tokens=False)
    length = spec["context_tokens"] - spec["max_new_tokens"]
    prompt = (pattern * (length // len(pattern) + 1))[:length]
    model.generate_ids(prompt[:32], min(8, spec["max_new_tokens"]), stop=False)
    timing = []
    for _ in range(2):
        output, stamps = model.generate_ids(prompt, spec["max_new_tokens"], stop=False)
        if len(stamps) < 2:
            raise ValueError("Insufficient decode steps")
        timing.append((len(stamps) - 1) / (stamps[-1] - stamps[0]))
    return {
        "documents": documents,
        "tasks": tasks,
        "task_accuracy": sum(t["correct"] for t in tasks) / len(tasks),
        "mean_nll": sum(d["nll"] for d in documents) / len(documents),
        "bits_per_byte": sum(d["bits_per_byte"] for d in documents) / len(documents),
        "decode_tps": min(timing),
        "decode_trials": timing,
        "seconds": time.perf_counter() - started,
        "capacity_context_tokens": len(prompt) + len(output),
        "concurrency": 1,
        "mlx_free_cache_limit_bytes": 256 * 2**20 if backend == "mlx" else None,
        "weight_file_bytes": sum(p.stat().st_size for p in Path(path).glob("*.safetensors")),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["build", "evaluate"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--candidate", required=True)
    p.add_argument("--split", default="development")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    spec, catalog, plan = [
        read(args.root / name) for name in ["spec.json", "catalog.json", "plan.json"]
    ]
    hosts = {h["id"]: h for h in catalog["hosts"]}
    candidate = next(c for c in plan["candidates"] if c["id"] == args.candidate)
    if args.stage == "build":
        result = {"path": str(build(args.root, candidate, hosts).resolve())}
    else:
        host = hosts[candidate["host"]]
        path = (
            Path(host["path"])
            if candidate["operation"] == "original"
            else args.root / "artifacts" / candidate["id"]
        )
        result = evaluate(
            path,
            host["backend"],
            read(args.root / f"{args.split}.json"),
            spec,
            host.get("tokenizer"),
        )
    write(args.output, result)


if __name__ == "__main__":
    main()
