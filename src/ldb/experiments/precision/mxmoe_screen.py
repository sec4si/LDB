"""Development-only final-output sensitivity in the upstream quantization format."""

import argparse
import json
import time
from pathlib import Path

import torch

from .helpers import evaluate, load, sha, split_batches, summarize, write
from .mxmoe_bridge import Bridge
from .mxmoe_model import PaperBank, convert
from .precision_baseline import MODEL, tokenizer


def replace(bank, expert, matrices):
    for part, matrix in zip(("gu", "down"), matrices):
        packed, scale, n, k, bits = matrix
        setattr(bank, f"{part}_{expert}_packed", packed)
        setattr(bank, f"{part}_{expert}_scale", scale)
        bank.shapes[part, expert] = (n, k, bits)


def select_blocks(rows, budget, exact=True):
    states = {0: (0.0, [])}
    for i, row in enumerate(rows):
        cost = 2 if row["part"] == "gu" else 1
        for used, (score, items) in list(states.items()):
            target = used + cost
            if target <= budget and (
                target not in states or score + row["gain"] > states[target][0]
            ):
                states[target] = (score + row["gain"], items + [i])
    chosen = states[budget] if exact else max(states.values(), key=lambda state: state[0])
    return [rows[i] for i in chosen[1]]


def run(source, output, library, blocks=False):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    write(
        output / "freeze.json",
        dict(
            development_sha256=sha(source / "development.json"),
            library_sha256=sha(library),
            protected_experts=8,
            block_allocation=blocks,
            upgrade_units=24,
            sources={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            scope="Development-only global KL restoration; upstream per-channel RTN; fixed eight-expert budget",
        ),
    )
    batches = split_batches(
        tokenizer(), json.loads((source / "development.json").read_text())["texts"], 128
    )
    batches = [{k: v.cuda() for k, v in b.items()} for b in batches]
    model = load(MODEL).cuda()
    bridge = Bridge(library)
    references, original = evaluate(model, batches)
    references = [r.cpu() for r in references]
    kernel = next(
        i for i, n in enumerate(bridge.names) if "w4a16_g-1_asym:" in n and "w8a16_g-1_asym:" in n
    )
    highs = [
        PaperBank(
            layer_row.mlp.experts, [8] * layer_row.mlp.experts.gate_up_proj.shape[0], bridge, kernel
        )
        for layer_row in model.model.layers
    ]
    plan = [[4] * h.num_experts for h in highs]
    convert(model, plan, bridge)

    def measure():
        rows = []
        for batch, ref in zip(batches, references, strict=True):
            _, r = evaluate(model, [batch], [ref.cuda()])
            rows.extend(r)
        return summarize(rows, original)

    started = time.perf_counter()
    baseline = measure()
    ranking = []
    for li, layer in enumerate(model.model.layers):
        bank = layer.mlp.experts
        for e in range(bank.num_experts):
            low = [bank.matrix(p, e) for p in ("gu", "down")]
            for part in ("gu", "down") if blocks else ("both",):
                replace(
                    bank,
                    e,
                    [
                        highs[li].matrix(p, e) if part in (p, "both") else low[i]
                        for i, p in enumerate(("gu", "down"))
                    ],
                )
                quality = measure()
                replace(bank, e, low)
                ranking.append(
                    dict(
                        layer=li,
                        expert=e,
                        part=part,
                        gain=baseline["mean_kl"] - quality["mean_kl"],
                        quality=quality,
                    )
                )
        write(output / "partial.json", dict(baseline=baseline, ranking=ranking))
        print("layer", li, "done", flush=True)
    ranking.sort(key=lambda r: (-r["gain"], r["layer"], r["expert"]))
    if blocks:
        plan = [[dict(gu=4, down=4) for _ in layer] for layer in plan]
    for row in select_blocks(ranking, 24) if blocks else ranking[:8]:
        li, e = row["layer"], row["expert"]
        bank = model.model.layers[li].mlp.experts
        if blocks:
            plan[li][e][row["part"]] = 8
        else:
            plan[li][e] = 8
        replace(
            bank,
            e,
            [
                highs[li].matrix(p, e) if row["part"] in (p, "both") else bank.matrix(p, e)
                for p in ("gu", "down")
            ],
        )
    write(
        output / "choice.json",
        dict(
            plan=plan,
            ranking=ranking,
            baseline=baseline,
            development_quality=measure(),
            seconds=time.perf_counter() - started,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "output", "library"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--blocks", action="store_true")
    a = p.parse_args()
    run(a.source, a.output, a.library, a.blocks)
