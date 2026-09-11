"""Lazy entry points for the optional research adapters."""

from __future__ import annotations

import subprocess
import sys
from typing import Annotated

import typer

experiment = typer.Typer(no_args_is_help=True, help="Run scoped MoE experiments.")
TOOLS = {
    "fingerprint": ("response", "CPU response distances and routing-weighted donor selection"),
    "ir": ("moe_ir", "Validate, inspect and compare MoE IR; import audited Mixtral studies"),
    "budget": (
        "build_budget_model",
        "Frozen budget plan; execution requires macOS footprint monitor",
    ),
    "expert-capture": ("expert_abi.capture", "Capture expert responses for the audited eager bank"),
    "expert-evaluate": ("expert_abi.evaluate", "Evaluate reversible expert replacements"),
    "expert-package": ("expert_abi.package", "Export and verify hash-bound expert packages"),
    "precision-build": (
        "precision.mxmoe_qwen_build",
        "Build packed Qwen weights with explicit native library",
    ),
    "precision-gptq": ("precision.qwen_gptq_build", "Calibrate and build GPTQ Qwen weights"),
    "precision-profile": (
        "precision.qwen_mixed_loss_profile",
        "Profile local mixed-precision loss",
    ),
    "precision-plan": (
        "precision.qwen_mixed_loss_plan",
        "Allocate gate/up and down precision from a profile",
    ),
    "precision-linear-plan": (
        "precision.qwen_linear_loss_plan",
        "Allocate gate, up and down independently",
    ),
    "precision-ppl": (
        "precision.qwen_wikitext_ppl",
        "Measure document PPL using explicit test tokens",
    ),
    "precision-bench": ("precision.qwen_activation_pack_bench", "Benchmark the packed executor"),
    "precision-check": ("precision.qwen_gptq_check", "Check restored packed checkpoints"),
    "route-capture": ("precision.qwen_route_capture", "Capture Qwen routing for replay"),
}


@experiment.command("list")
def list_tools() -> None:
    """List adapters without importing any model backend."""
    for name, (_, description) in TOOLS.items():
        typer.echo(f"{name}: {description}")


@experiment.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, name: Annotated[str, typer.Argument()]) -> None:
    """Run NAME with its original arguments: run NAME -- --help."""
    if name not in TOOLS:
        raise typer.BadParameter(f"Unknown experiment: {name}; use 'ldb experiment list'")
    arguments = list(ctx.args)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    result = subprocess.run(
        [sys.executable, "-m", f"ldb.experiments.{TOOLS[name][0]}", *arguments], check=False
    )
    raise typer.Exit(result.returncode)
