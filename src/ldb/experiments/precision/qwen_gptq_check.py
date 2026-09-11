"""Validate restored GPTQ Qwen against its saved layer-sequential forward."""

import argparse
from pathlib import Path

import torch

from .helpers import sha, write
from .mxmoe_qwen_restore import restore


@torch.inference_mode()
def run(checkpoint, library, output, offload=False):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = restore(checkpoint, library, device="cpu" if offload else "cuda")
    for layer in model.model.layers:
        layer.mlp.experts.grouped_routing = True
    reference = torch.load(checkpoint / "validation.pt", weights_only=True)
    if offload:
        from .qwen_wikitext_ppl import native_hidden

        state = next(native_hidden(model, [reference["input_ids"]]))
        model.lm_head.cuda()
        actual = model.lm_head(state).cpu()
    else:
        actual = model(
            input_ids=reference["input_ids"].cuda(), use_cache=False, output_router_logits=False
        ).logits.cpu()
    expected = reference["logits"]
    report = dict(
        exact_logits=torch.equal(actual, expected),
        max_abs_error=float((actual - expected).abs().max()),
        checkpoint_sha256=sha(checkpoint / "state.pt"),
        validation_sha256=sha(checkpoint / "validation.pt"),
        library_sha256=sha(library),
        offload=offload,
        script_sha256=sha(Path(__file__)),
        scope="Restored layer-offloaded forward versus saved sequential forward"
        if offload
        else "Restored full forward versus saved sequential quantized-layer forward",
    )
    write(output, report)
    print(report, flush=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "library", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--offload", action="store_true")
    run(**vars(p.parse_args()))
