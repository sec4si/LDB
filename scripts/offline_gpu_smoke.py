"""Validate offline debugging on a real CUDA device using a local GPT-2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ldb.adapters.transformers import TransformersAdapter
from ldb.stepping import LayerDebugger
from ldb.structural import structural_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback does not qualify")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    adapter = TransformersAdapter(args.model, device="cuda")
    inputs = adapter.tokenizer("The Eiffel Tower is in", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        expected = adapter.model(**inputs, use_cache=False).logits
    debugger = LayerDebugger(adapter.model, inputs)
    debugger.command("break 0 norm 0")
    assert debugger.run()["stop_reason"] == "breakpoint"
    debugger.save(root / "middle.safetensors")
    delta = debugger.branch(1, 0)["max_logit_delta"]
    assert delta > 0
    debugger.run()
    assert torch.equal(debugger.logits(), expected)
    replay = LayerDebugger(adapter.model, inputs)
    replay.load(root / "middle.safetensors")
    replay.run()
    assert torch.equal(replay.logits(), expected)
    width = adapter.model.config.n_embd
    dictionary = root / "synthetic-sae.safetensors"
    save_file(
        {
            "encoder": torch.zeros(width, 1),
            "decoder": torch.zeros(1, width),
            "encoder_bias": torch.ones(1),
        },
        dictionary,
    )
    replay.restore("start")
    replay.watch_feature("synthetic", 0, dictionary, 0, 0.5)
    assert replay.run()["stop_reason"] == "watch"
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "stepping_matches_forward": True,
        "disk_replay_exact": True,
        "conditional_breakpoint": True,
        "synthetic_feature_watch": True,
        "branch_max_logit_delta": delta,
        "structural": structural_experiment(
            args.model, device="cuda", prompt="The Eiffel Tower is in", layer=0
        ),
    }
    content = json.dumps(result, indent=2)
    (root / "result.json").write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
