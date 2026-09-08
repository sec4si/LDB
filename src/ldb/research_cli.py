"""Offline entry points for weight export, training gradients, and operator stepping."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

console = Console()
weights = typer.Typer(no_args_is_help=True, help="Inspect and export permanent weight edits.")
research = typer.Typer()


def load_model(model: str, revision: str | None, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    factory: Any = AutoModelForCausalLM
    loaded = (
        factory.from_pretrained(model, revision=revision, dtype="auto", attn_implementation="eager")
        .to(device)
        .eval()
    )
    return loaded, tokenizer


@weights.command("list")
def list_weights(
    model: Annotated[str, typer.Argument()],
    pattern: Annotated[str, typer.Option()] = "*",
    revision: Annotated[str | None, typer.Option()] = None,
) -> None:
    """List actual parameter names and shapes, including tied aliases."""
    from ldb.gradients import select_names

    loaded, _ = load_model(model, revision, "cpu")
    parameters = dict(loaded.named_parameters(remove_duplicate=False))
    console.print_json(
        data={
            name: {"shape": list(parameters[name].shape), "dtype": str(parameters[name].dtype)}
            for name in select_names(list(parameters), [pattern])
        }
    )


@weights.command("export")
def export_weights(
    model: Annotated[str, typer.Argument()],
    output: Annotated[Path, typer.Option()],
    parameter: Annotated[str, typer.Option()],
    operation: Annotated[str, typer.Option()] = "scale",
    value: Annotated[float | None, typer.Option()] = None,
    tensor: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    axis: Annotated[int | None, typer.Option()] = None,
    start: Annotated[int | None, typer.Option()] = None,
    stop: Annotated[int | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "cpu",
    revision: Annotated[str | None, typer.Option()] = None,
    max_shard_size: Annotated[str, typer.Option()] = "2GB",
) -> None:
    """Save a new reloadable model; the source directory and in-memory weights are preserved."""
    from ldb.weight_editing import export_weight_edit

    if output.exists():
        raise typer.BadParameter("Output must be a new directory")
    loaded, tokenizer = load_model(model, revision, device)
    report = export_weight_edit(
        loaded,
        output,
        parameter=parameter,
        operation=operation,
        value=value,
        tensor=tensor,
        axis=axis,
        start=start,
        stop=stop,
        tokenizer=tokenizer,
        max_shard_size=max_shard_size,
    )
    console.print_json(data={"output": str(output), "edit": report})


@research.command("gradients")
def gradients(
    model: Annotated[str, typer.Argument()],
    text: Annotated[list[str], typer.Option("--text")],
    output: Annotated[Path, typer.Option()],
    parameter: Annotated[list[str] | None, typer.Option("--parameter")] = None,
    module: Annotated[list[str] | None, typer.Option("--module")] = None,
    steps: Annotated[int, typer.Option(min=1, max=100)] = 1,
    learning_rate: Annotated[float, typer.Option(min=0)] = 0.0,
    device: Annotated[str, typer.Option()] = "cpu",
    revision: Annotated[str | None, typer.Option()] = None,
    capture: Annotated[bool, typer.Option()] = False,
    norm_threshold: Annotated[float | None, typer.Option(min=0)] = None,
    max_events: Annotated[int, typer.Option(min=1)] = 2000,
) -> None:
    """Trace bounded causal-LM backward steps; a positive learning rate enables SGD updates."""
    import math

    import torch

    from ldb.gradients import GradientTracer

    if output.exists():
        raise typer.BadParameter("Output must be a new directory")
    if not math.isfinite(learning_rate):
        raise typer.BadParameter("Learning rate must be finite")
    torch.manual_seed(0)
    loaded, tokenizer = load_model(model, revision, device)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    labels = inputs["input_ids"].masked_fill(inputs["attention_mask"] == 0, -100)
    if labels.shape[1] < 2 or not bool((labels[:, 1:] != -100).any()):
        raise typer.BadParameter("Text must contain at least two tokens for causal-LM loss")
    loaded.train()
    optimizer = torch.optim.SGD(loaded.parameters(), lr=learning_rate) if learning_rate else None
    tracer = GradientTracer(
        loaded,
        parameters=parameter or ["*"],
        modules=module or [],
        capture=capture,
        norm_threshold=norm_threshold,
        max_events=max_events,
    )
    metadata: dict[str, Any] = {
        "model": model,
        "revision": getattr(loaded.config, "_commit_hash", None),
        "device": device,
        "learning_rate": learning_rate,
        "optimizer": "SGD" if optimizer else None,
        "losses": [],
        "gradient_states": [],
        "status": "running",
        "optimizer_updates": 0,
        "input_ids": inputs["input_ids"].cpu().tolist(),
    }
    try:
        with tracer:
            for index in range(steps):
                tracer.mark_step(index)
                loaded.zero_grad(set_to_none=True)
                loss = loaded(**inputs, labels=labels, use_cache=False).loss
                tracer.record("loss", "loss", loss)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite loss; optimizer update refused")
                metadata["losses"].append(float(loss.detach()))
                loss.backward()
                metadata["gradient_states"].append(tracer.snapshot("after_backward"))
                if any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in loaded.parameters()
                ):
                    raise RuntimeError("Nonfinite gradient; optimizer update refused")
                if optimizer is not None:
                    optimizer.step()
                    metadata["optimizer_updates"] += 1
        metadata["status"] = "completed"
    except BaseException as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        tracer.save(output, metadata=metadata)
    console.print_json(
        data={
            "output": str(output),
            "events": len(tracer.events),
            "dropped_events": tracer.dropped_events,
            **metadata,
        }
    )


@research.command("debug-ops")
def debug_operators(
    model: Annotated[str, typer.Argument()],
    prompt: Annotated[str, typer.Option()] = "The Eiffel Tower is in",
    device: Annotated[str, typer.Option()] = "cpu",
    revision: Annotated[str | None, typer.Option()] = None,
    backward: Annotated[bool, typer.Option()] = False,
    max_operations: Annotated[int, typer.Option(min=1)] = 100000,
    timeout: Annotated[float, typer.Option(min=0.1)] = 30.0,
    execute: Annotated[list[str] | None, typer.Option("--execute", "-e")] = None,
) -> None:
    """Pause and step visible ATen operators in one eager forward (optionally backward)."""
    import torch

    from ldb.operators import OperatorDebugger, describe

    loaded, tokenizer = load_model(model, revision, device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if backward and inputs["input_ids"].shape[1] < 2:
        raise typer.BadParameter("Backward requires at least two tokens")

    def run() -> Any:
        if backward:
            output = loaded(**inputs, labels=inputs["input_ids"], use_cache=False)
            output.loss.backward()
            return output.loss.detach()
        with torch.no_grad():
            return loaded(**inputs, use_cache=False).logits

    with OperatorDebugger(run, timeout=timeout, max_operations=max_operations) as debugger:
        console.print_json(data=debugger.inspect())
        if execute is not None:
            for command in execute:
                console.print_json(data={"command": command, "result": debugger.command(command)})
        else:
            console.print(
                "Operator debugger: step, continue, inspect, history, break/delete PATTERN, quit"
            )
            while debugger.inspect()["state"] == "paused":
                try:
                    command = input("(ldb-op) ")
                    if command.strip() in {"quit", "exit"}:
                        break
                    console.print_json(data=debugger.command(command))
                except (EOFError, KeyboardInterrupt):
                    break
                except (ValueError, OSError) as error:
                    console.print(f"Error: {error}")
        if debugger.inspect()["state"] == "finished":
            console.print_json(data={"result": describe(debugger.result)})
