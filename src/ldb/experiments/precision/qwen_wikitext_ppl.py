"""WikiText2 PPL with upstream segmentation and bounded cross-entropy workspace."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from .helpers import sha, write
from .mxmoe_qwen_restore import restore
from .qwen_gptq_build import forward_layer


def sequence_loss(logits, ids, chunk=128):
    # Preserve full-sequence LM-head GEMM shape; chunk only independent CE rows.
    losses = []
    for start in range(0, ids.shape[1] - 1, chunk):
        stop = min(start + chunk, ids.shape[1] - 1)
        losses.append(
            F.cross_entropy(logits[0, start:stop], ids[0, start + 1 : stop + 1], reduction="none")
        )
    return torch.cat(losses).mean()


@torch.inference_mode()
def native_hidden(model, ids):
    model.model.embed_tokens.cuda()
    states = [model.model.embed_tokens(x.cuda()).cpu() for x in ids]
    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cuda()
    for i, layer in enumerate(model.model.layers):
        layer.cuda()
        states = [forward_layer(model, layer, x) for x in states]
        layer.cpu()
        print("reference layer", i, flush=True)
    model.model.norm.cuda()
    for state in states:
        yield model.model.norm(state.cuda())


@torch.inference_mode()
def run(
    source,
    test_tokens,
    output,
    checkpoint=None,
    library=None,
    input_len=4096,
    limit=None,
    offload=False,
    fused_activation_pack=False,
    coalesce_metadata=False,
):
    if (fused_activation_pack or coalesce_metadata) and checkpoint is None:
        raise ValueError("Fused activation packing requires a mixed checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens = torch.tensor(json.loads(test_tokens.read_text())["input_ids"], dtype=torch.long)
    available = tokens.numel() // input_len
    count = available if limit is None else min(limit, available)
    if not count:
        raise ValueError("No complete test segments")
    ids = [tokens[:, i * input_len : (i + 1) * input_len] for i in range(count)]
    freeze = dict(
        test_sha256=sha(test_tokens),
        input_len=input_len,
        segments=count,
        model_manifest_sha256=sha(source / "download-complete.json"),
        available_segments=available,
        discarded_tail_tokens=tokens.numel() % input_len,
        complete_test=limit is None or count == available,
        packed_layer_offload=offload,
        fused_activation_pack=fused_activation_pack,
        coalesce_metadata=coalesce_metadata,
        sources={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        scope="Upstream contiguous segmentation and mean sequence NLL; full LM-head GEMM, chunked CE rows; not task accuracy",
    )
    if checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            dtype=torch.float16,
            attn_implementation="eager",
            experts_implementation="eager",
        ).eval()
        hidden = native_hidden(model, ids)
    else:
        model = restore(checkpoint, library, device="cpu" if offload else "cuda")
        if fused_activation_pack or coalesce_metadata:
            from .mxmoe_activation_bank import ActivationBank
            from .mxmoe_linear_bank import LinearBank

            banks = [layer.mlp.experts for layer in model.model.layers]
            if not all(isinstance(bank, (ActivationBank, LinearBank)) for bank in banks):
                raise ValueError("Fused activation packing requires mixed activation banks")
            for bank in banks:
                bank.fused_activation_pack = fused_activation_pack
                bank.coalesce_metadata = coalesce_metadata
        for layer in model.model.layers:
            layer.mlp.experts.grouped_routing = True
        hidden = (
            native_hidden(model, ids)
            if offload
            else (model.model(input_ids=x.cuda(), use_cache=False).last_hidden_state for x in ids)
        )
        freeze.update(checkpoint_sha256=sha(checkpoint / "state.pt"), library_sha256=sha(library))
    write(output / "freeze.json", freeze)
    rows = []
    losses = []
    for i, (state, x) in enumerate(zip(hidden, ids, strict=True)):
        model.lm_head.cuda()
        logits = model.lm_head(state)
        loss = sequence_loss(logits, x.cuda()).float()
        losses.append(loss.cpu() * input_len)
        rows.append(dict(segment=i, nll=loss.item()))
        write(output / "partial.json", rows)
        print(rows[-1], flush=True)
        del logits
    ppl = torch.exp(torch.stack(losses).sum() / (count * input_len)).item()
    write(
        output / "result.json", dict(ppl=ppl, documents=rows, complete_test=freeze["complete_test"])
    )
    print("PPL", ppl, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "test-tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint", "library"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--input-len", type=int, default=4096)
    p.add_argument("--limit", type=int)
    p.add_argument("--offload", action="store_true")
    p.add_argument("--fused-activation-pack", action="store_true")
    p.add_argument("--coalesce-metadata", action="store_true")
    run(**vars(p.parse_args()))
