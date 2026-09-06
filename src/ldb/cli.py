"""Command-line interface for LDB."""

from __future__ import annotations

import fnmatch
import shlex
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ldb import __version__
from ldb.causal import causal_trace
from ldb.client import AgentGroupClient
from ldb.models import HookPoint, RunManifest
from ldb.patches import load_patch
from ldb.runtime import ProbeCallback
from ldb.search import load_suite, search_scale_patches
from ldb.trace import RunBundleRecorder, load_events, load_manifest

app = typer.Typer(no_args_is_help=True, help="Debug latent state in large language models.")
patch_app = typer.Typer(no_args_is_help=True, help="Validate and manage Neural Patches.")
app.add_typer(patch_app, name="patch")
console = Console()

_ATTACH_HELP = """Commands:
  status
  arm [--capture] [--request REQUEST_ID] [HOOK_GLOB ...]
  disarm
  events [AFTER_SEQUENCE]
  clear
  patch-set PATCH_ID HOOK OP [VALUE] [--tensor PATH] [--expert INDEX] [--request REQUEST_ID]
  patch-remove PATCH_ID
  patches
  break BREAKPOINT_ID HOOK [--request REQUEST_ID] [--timeout SECONDS]
  delete BREAKPOINT_ID
  breaks
  continue [HIT_ID]
  help
  quit
"""


def _record_callback(recorder: RunBundleRecorder, *, capture: bool) -> ProbeCallback:
    def callback(hook: HookPoint, tensor: Any) -> None:
        recorder.record(hook, tensor, capture=capture)

    return callback


@app.command()
def version() -> None:
    """Print the LDB version."""

    console.print(__version__)


@app.command()
def run(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    prompt: Annotated[str, typer.Option("--prompt", help="Prompt text")],
    output: Annotated[Path, typer.Option("--output", help="RunBundle parent directory")] = Path(
        "runs"
    ),
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = 1,
    capture: Annotated[
        list[str] | None,
        typer.Option("--capture", help="Glob selecting hooks whose full tensors are persisted"),
    ] = None,
) -> None:
    """Run deterministic inference and write a latent trace bundle."""

    from ldb.adapters.transformers import TransformersAdapter

    captures = capture or []
    adapter = TransformersAdapter(model, revision=revision, device=device)
    with RunBundleRecorder(output) as recorder:
        handles = []
        for hook in adapter.hook_points():
            selected = any(fnmatch.fnmatchcase(hook.name, pattern) for pattern in captures)
            handles.append(
                adapter.install_probe(
                    hook,
                    _record_callback(recorder, capture=selected),
                )
            )
        try:
            output_text = adapter.generate(prompt, max_new_tokens=max_new_tokens)
        finally:
            for handle in handles:
                handle.remove()

        manifest = RunManifest(
            run_id=recorder.run_id,
            model_id=model,
            model_revision=revision,
            tokenizer_id=model,
            backend="transformers",
            backend_version=adapter.backend_version,
            adapter="ldb.adapters.transformers",
            adapter_version=adapter.adapter_version,
            prompt=prompt,
            prompt_token_ids=tuple(adapter.tokenize(prompt)),
            generation={"max_new_tokens": max_new_tokens, "do_sample": False},
            output_text=output_text,
        )
        recorder.write_manifest(manifest)

    console.print(f"RunBundle: [bold]{recorder.path}[/bold]")
    console.print(f"Output: {output_text}")


@app.command()
def inspect(bundle: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Inspect a recorded RunBundle."""

    manifest = load_manifest(bundle)
    events = load_events(bundle)
    console.print(f"[bold]{manifest.run_id}[/bold]  {manifest.model_id}  {manifest.backend}")
    console.print(f"Prompt: {manifest.prompt}")
    console.print(f"Output: {manifest.output_text or '<incomplete>'}")

    table = Table("Seq", "Hook", "Shape", "Norm", "Captured")
    for event in events:
        table.add_row(
            str(event.sequence),
            event.hook,
            " × ".join(str(size) for size in event.summary.shape),
            f"{event.summary.norm:.5g}" if event.summary.norm is not None else "-",
            "yes" if event.tensor_file else "no",
        )
    console.print(table)


@app.command()
def causal(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    clean_prompt: Annotated[str, typer.Option("--clean-prompt")],
    corrupt_prompt: Annotated[str, typer.Option("--corrupt-prompt")],
    target_token_id: Annotated[int, typer.Option("--target-token-id", min=0)],
    hook: Annotated[
        list[str] | None,
        typer.Option("--hook", help="Glob selecting causal patch candidates"),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Localize a next-token behavior with clean/corrupt activation patching."""

    from ldb.adapters.transformers import TransformersAdapter

    adapter = TransformersAdapter(model, revision=revision, device=device)
    report = causal_trace(
        adapter,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        target_token_id=target_token_id,
        hook_globs=hook or ["block.*.residual.post"],
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    table = Table("Hook", "Patched log P", "Effect", "Recovery")
    for trial in report.trials:
        table.add_row(
            trial.hook,
            f"{trial.patched_log_probability:.6g}",
            f"{trial.effect:+.6g}",
            f"{trial.recovery:.2%}" if trial.recovery is not None else "-",
        )
    console.print(
        f"Clean log P: {report.clean_log_probability:.6g}  "
        f"Corrupt log P: {report.corrupt_log_probability:.6g}"
    )
    console.print(table)
    if output is not None:
        console.print(f"Report: [bold]{output}[/bold]")


@app.command("search")
def search_patches(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    suite: Annotated[Path, typer.Option("--suite", exists=True, dir_okay=False)],
    hook: Annotated[
        list[str] | None,
        typer.Option("--hook", help="Glob selecting candidate hook points"),
    ] = None,
    scale: Annotated[
        list[float] | None,
        typer.Option("--scale", help="Candidate activation scale"),
    ] = None,
    regression_weight: Annotated[float, typer.Option("--regression-weight", min=0.0)] = 1.0,
    complexity_weight: Annotated[float, typer.Option("--complexity-weight", min=0.0)] = 0.01,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Search minimal scalar activation patches with regression penalties."""

    from ldb.adapters.transformers import TransformersAdapter

    adapter = TransformersAdapter(model, revision=revision, device=device)
    report = search_scale_patches(
        adapter,
        load_suite(suite),
        hook_globs=hook or ["block.*.residual.post"],
        scales=scale or [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        regression_weight=regression_weight,
        complexity_weight=complexity_weight,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    table = Table("Hook", "Scale", "Target gain", "Regression loss", "Objective")
    for trial in report.trials[:20]:
        table.add_row(
            trial.hook,
            f"{trial.scale:.6g}",
            f"{trial.target_gain:+.6g}",
            f"{trial.regression_loss:.6g}",
            f"{trial.objective:+.6g}",
        )
    console.print(table)
    if output is not None:
        console.print(f"Report: [bold]{output}[/bold]")


def _run_agent_command(client: AgentGroupClient, command: str) -> bool:
    parts = shlex.split(command)
    if not parts:
        return True
    name, *arguments = parts
    if name in {"quit", "exit"}:
        return False
    if name == "help":
        console.print(_ATTACH_HELP)
        return True

    method: str
    params: dict[str, Any] = {}
    if name == "status":
        method = "status"
    elif name == "arm":
        capture, hooks, request_ids = _parse_arm_arguments(arguments)
        method, params = (
            "arm",
            {
                "hooks": hooks,
                "request_ids": request_ids,
                "capture_tensors": capture,
            },
        )
    elif name == "disarm":
        method = "disarm"
    elif name == "events":
        method = "events"
        if arguments:
            params["after"] = int(arguments[0])
    elif name == "clear":
        method = "clear"
    elif name == "patches":
        method = "patch.list"
    elif name == "patch-remove" and len(arguments) == 1:
        method, params = "patch.remove", {"patch_id": arguments[0]}
    elif name == "patch-set" and len(arguments) >= 3:
        patch_id, hook, operation = arguments[:3]
        value, tensor, expert, request_ids = _parse_patch_arguments(arguments[3:])
        params = {
            "patch_id": patch_id,
            "hook": hook,
            "operation": operation,
            "request_ids": request_ids,
        }
        if value is not None:
            params["value"] = value
        if tensor is not None:
            params["tensor"] = tensor
        if expert is not None:
            params["expert"] = expert
        method = "patch.set"
    elif name == "break" and len(arguments) >= 2:
        breakpoint_id, hook = arguments[:2]
        request_ids, timeout = _parse_break_arguments(arguments[2:])
        params = {
            "breakpoint_id": breakpoint_id,
            "hook": hook,
            "request_ids": request_ids,
            "timeout_seconds": timeout,
        }
        method = "break.set"
    elif name == "delete" and len(arguments) == 1:
        method, params = "break.remove", {"breakpoint_id": arguments[0]}
    elif name == "breaks":
        method = "break.list"
    elif name == "continue" and len(arguments) <= 1:
        method = "continue"
        if arguments:
            params["hit_id"] = arguments[0]
    else:
        raise ValueError(f"Invalid attach command: {command!r}. Type 'help' for usage.")

    results = client.call(method, params)
    console.print_json(data=results)
    return True


def _parse_arm_arguments(arguments: list[str]) -> tuple[bool, list[str], list[str]]:
    capture = False
    hooks: list[str] = []
    request_ids: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--capture":
            capture = True
        elif value == "--request" and index + 1 < len(arguments):
            index += 1
            request_ids.append(arguments[index])
        elif value.startswith("--"):
            raise ValueError(f"Unknown arm option: {value}")
        else:
            hooks.append(value)
        index += 1
    return capture, hooks or ["*"], request_ids


def _parse_patch_arguments(
    arguments: list[str],
) -> tuple[float | None, str | None, int | None, list[str]]:
    value: float | None = None
    tensor: str | None = None
    expert: int | None = None
    request_ids: list[str] = []
    index = 0
    if arguments and not arguments[0].startswith("--"):
        value = float(arguments[0])
        index = 1
    while index < len(arguments):
        option = arguments[index]
        if option not in {"--request", "--tensor", "--expert"} or index + 1 >= len(arguments):
            raise ValueError(f"Invalid patch option: {option}")
        index += 1
        if option == "--request":
            request_ids.append(arguments[index])
        elif option == "--tensor":
            if tensor is not None:
                raise ValueError("--tensor may only be specified once")
            tensor = arguments[index]
        else:
            if expert is not None:
                raise ValueError("--expert may only be specified once")
            expert = int(arguments[index])
        index += 1
    return value, tensor, expert, request_ids


def _parse_break_arguments(arguments: list[str]) -> tuple[list[str], float]:
    request_ids: list[str] = []
    timeout = 300.0
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if option not in {"--request", "--timeout"} or index + 1 >= len(arguments):
            raise ValueError(f"Invalid breakpoint option: {option}")
        index += 1
        if option == "--request":
            request_ids.append(arguments[index])
        else:
            timeout = float(arguments[index])
        index += 1
    return request_ids, timeout


@app.command()
def attach(
    target: Annotated[
        Path,
        typer.Argument(exists=True, help="Agent socket or directory containing agent descriptors"),
    ],
    execute: Annotated[
        str | None,
        typer.Option("--execute", "-e", help="Execute one debugger command and exit"),
    ] = None,
) -> None:
    """Attach to one or more live LDB runtime agents."""

    client = AgentGroupClient.connect(target)
    hello = client.call("hello")
    console.print_json(data=hello)
    if execute is not None:
        _run_agent_command(client, execute)
        return

    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    console.print("Type 'help' for commands.")
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    while True:
        try:
            command = session.prompt("(ldb) ")
            if not _run_agent_command(client, command):
                return
        except (EOFError, KeyboardInterrupt):
            return
        except (OSError, RuntimeError, ValueError) as exc:
            console.print(f"[red]error:[/red] {exc}")


@patch_app.command("validate")
def validate_patch(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate a Neural Patch YAML document."""

    patch = load_patch(path)
    console.print(
        f"Valid {patch.schema_version}: [bold]{patch.patch_id}[/bold] "
        f"({len(patch.interventions)} intervention(s))"
    )


if __name__ == "__main__":
    app()
