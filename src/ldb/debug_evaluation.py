"""Replay next-token regression suites with a named debugger edit."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ldb.search import load_suite

if TYPE_CHECKING:
    from ldb.stepping import LayerDebugger


def evaluate_debugger_suite(debugger: LayerDebugger, path: Path, command: str) -> dict[str, Any]:
    from ldb.stepping import LayerDebugger

    if debugger.tokenizer is None:
        raise ValueError("Regression suites require a tokenizer")
    suite = load_suite(path)
    rows = []
    device = next(debugger.model.parameters()).device
    for group in ("target", "related", "regression"):
        for case in getattr(suite, group):
            inputs = debugger.tokenizer(case.prompt, return_tensors="pt").to(device)
            token = case.target_token_id
            if token is None:
                prompt_ids = inputs["input_ids"][0].tolist()
                combined = debugger.tokenizer.encode(case.prompt + case.target_text)
                if (
                    combined[: len(prompt_ids)] != prompt_ids
                    or len(combined) != len(prompt_ids) + 1
                ):
                    raise ValueError(
                        "Debugger regression requires one aligned next token per case; "
                        "use ldb patch verify for multi-token continuations"
                    )
                token = combined[-1]
            replay = LayerDebugger(debugger.model, inputs, tokenizer=debugger.tokenizer)
            replay._restore_patches(debugger._patch_state())
            replay._features.update(replay._validate_edits(replay.edits))
            replay.target(token, (token + 1) % debugger.model.config.vocab_size)
            comparison = replay.compare(command)
            baseline, patched = comparison["baseline"], comparison["patched"]
            before = baseline["answer_log_probability"][0]
            after = patched["answer_log_probability"][0]
            rows.append(
                {
                    "case_id": case.case_id,
                    "group": group,
                    "category": case.category,
                    "weight": case.weight,
                    "target_token_id": token,
                    "baseline_log_probability": before,
                    "patched_log_probability": after,
                    "delta": after - before,
                    "baseline_correct": baseline["next_token_ids"][0] == token,
                    "patched_correct": patched["next_token_ids"][0] == token,
                }
            )
    groups = {}
    for group in ("target", "related", "regression"):
        selected = [row for row in rows if row["group"] == group]
        if selected:
            groups[group] = {
                "count": len(selected),
                "weighted_mean_delta": sum(row["delta"] * row["weight"] for row in selected)
                / sum(row["weight"] for row in selected),
                "worst_delta": min(row["delta"] for row in selected),
                "baseline_correct": sum(row["baseline_correct"] for row in selected),
                "patched_correct": sum(row["patched_correct"] for row in selected),
            }
    return {
        "command": command,
        "metric": "next-token log probability",
        "groups": groups,
        "cases": rows,
    }
