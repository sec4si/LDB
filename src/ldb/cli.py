"""Command-line interface for LDB."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ldb import __version__
from ldb.models import HookPoint, RunManifest
from ldb.patches import load_patch
from ldb.runtime import ProbeCallback
from ldb.trace import RunBundleRecorder, load_events, load_manifest

app = typer.Typer(no_args_is_help=True, help="Debug latent state in large language models.")
patch_app = typer.Typer(no_args_is_help=True, help="Validate and manage Neural Patches.")
app.add_typer(patch_app, name="patch")
console = Console()


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
