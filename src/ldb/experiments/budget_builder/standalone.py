"""Independent runner copied beside the delivered weights; no research imports."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("--task", choices=["general", "arithmetic", "extract"], default="general")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--receipt", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    contract = json.loads((root / "budget-model.json").read_text())
    for name, expected in contract["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(root / name) != expected:
            raise ValueError("Delivered artifact integrity failure")
    if not 1 <= a.max_tokens <= contract["spec"]["max_new_tokens"]:
        raise ValueError("Generation exceeds verified contract")
    from budget_prompts import messages

    chat = (
        [{"role": "user", "content": a.prompt}]
        if a.task == "general"
        else messages({"kind": a.task, "prompt": a.prompt})
    )
    if contract["backend"] == "mlx":
        import mlx.core as mx

        mx.set_cache_limit(256 * 2**20)
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        model, tok = load(str(root))
        prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if len(tok.encode(prompt)) + a.max_tokens > contract["spec"]["context_tokens"]:
            raise ValueError("Prompt exceeds verified context")
        answer = generate(
            model,
            tok,
            prompt=prompt,
            max_tokens=a.max_tokens,
            sampler=make_sampler(0),
            verbose=False,
        )
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            experts_implementation="eager",
        ).eval()
        tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
        prompt = (
            tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if getattr(tok, "chat_template", None)
            else "\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\nassistant:"
        )
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] + a.max_tokens > contract["spec"]["context_tokens"]:
            raise ValueError("Prompt exceeds verified context")
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=a.max_tokens, do_sample=False)
        answer = tok.decode(tokens[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(answer)
    if a.receipt:
        a.receipt.write_text(
            json.dumps(
                {"loaded": True, "answer": answer, "checked_files": len(contract["files"])},
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
