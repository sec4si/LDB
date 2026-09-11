"""Run the independently loadable local model and its frozen task interface."""

import argparse
import hashlib
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--task",
        choices=["general", "arithmetic", "sorting", "extract", "word"],
        default="general",
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    model, tokenizer = load(str(directory))
    if args.task == "general":
        messages = [{"role": "user", "content": args.prompt}]
    else:
        contract_path = directory / "prompt-contract.json"
        if not contract_path.exists():
            parser.error("This model directory does not include a calibrated task interface")
        contract = json.loads(contract_path.read_text())
        source = directory / "prompts.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != contract["prompt_source_sha256"]:
            parser.error("The task prompt implementation differs from its frozen contract")
        from prompts import messages as task_messages

        kind = ["arithmetic", "sorting", "extract", "word"].index(args.task)
        messages = task_messages(args.prompt, contract["style"], kind)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(0),
            verbose=False,
        )
    )


if __name__ == "__main__":
    main()
