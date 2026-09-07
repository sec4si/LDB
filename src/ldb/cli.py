"""Command-line interface for LDB."""

from __future__ import annotations

import fnmatch
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file

from ldb import __version__
from ldb.causal import causal_ablation, causal_propagation_graph, causal_trace
from ldb.client import AgentGroupClient
from ldb.identity import resolve_tokenizer_identity
from ldb.models import HookPoint, RunManifest, ValidationSummary
from ldb.patches import load_patch, write_patch
from ldb.runtime import ProbeCallback
from ldb.sae import FeatureReport, SAEDictionary, inspect_sae_features
from ldb.search import (
    load_candidate_set,
    load_suite,
    patch_from_combination_search,
    patch_from_search,
    search_candidate_combinations,
    search_scale_patches,
)
from ldb.trace import RunBundleRecorder, load_events, load_manifest
from ldb.verification import verify_patch

app = typer.Typer(no_args_is_help=True, help="Debug latent state in large language models.")
patch_app = typer.Typer(no_args_is_help=True, help="Validate and manage Neural Patches.")
app.add_typer(patch_app, name="patch")
console = Console()

_ATTACH_HELP = """Commands:
  status
  hooks [HOOK_GLOB]
  arm [--capture] [--request REQUEST_ID] [HOOK_GLOB ...]
  disarm
  events [AFTER_SEQUENCE]
  clear
  patch-set PATCH_ID HOOK OP [VALUE] [--tensor PATH] [--expert INDEX]
    [--head INDEX --heads COUNT] [--dictionary PATH --feature INDEX]
    [--token INDEX] [--request REQUEST_ID]
  patch-load PATH [--request REQUEST_ID]
  patch-remove PATCH_ID
  patches
  break BREAKPOINT_ID HOOK [--request REQUEST_ID] [--timeout SECONDS]
  delete BREAKPOINT_ID
  breaks
  inspect-hit HIT_ID [--capture]
  continue [HIT_ID]
  help
  quit
"""


@dataclass
class _PatchArguments:
    value: float | None = None
    tensor: str | None = None
    expert: int | None = None
    head: int | None = None
    num_heads: int | None = None
    dictionary: str | None = None
    feature: int | None = None
    token: int | None = None
    request_ids: list[str] = field(default_factory=list)


def _record_callback(recorder: RunBundleRecorder, *, capture: bool) -> ProbeCallback:
    def callback(hook: HookPoint, tensor: Any) -> None:
        recorder.record(hook, tensor, capture=capture)

    return callback


@app.command()
def version() -> None:
    """Print the LDB version."""

    console.print(__version__)


@app.command("identity")
def identity(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON identity")] = None,
) -> None:
    """Resolve an immutable model revision and canonical tokenizer hash."""

    resolved = resolve_tokenizer_identity(model, revision=revision)
    content = resolved.model_dump_json(indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    console.print_json(content)


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
            model_revision=adapter.revision,
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


@app.command("features")
def inspect_features(
    bundle: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    hook: Annotated[str, typer.Option("--hook", help="Captured normalized hook name")],
    dictionary_path: Annotated[Path, typer.Option("--dictionary", exists=True, dir_okay=False)],
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 20,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
) -> None:
    """Inspect top SAE features in a captured activation."""

    manifest = load_manifest(bundle)
    matches = [event for event in load_events(bundle) if event.hook == hook and event.tensor_file]
    if not matches:
        raise ValueError(f"RunBundle has no captured tensor for hook {hook!r}")
    event = matches[-1]
    tensor_file = event.tensor_file
    assert tensor_file is not None
    activation = load_file(bundle / tensor_file)["activation"]
    dictionary = SAEDictionary(dictionary_path)
    features = inspect_sae_features(activation, dictionary, top_k=top_k)
    report = FeatureReport(
        run_id=manifest.run_id,
        hook=hook,
        sequence=event.sequence,
        dictionary_hash=dictionary.digest,
        token_count=int(activation.numel() // dictionary.model_width),
        feature_count=dictionary.feature_count,
        features=features,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    table = Table("Feature", "Peak", "Mean", "Active")
    for feature in features:
        table.add_row(
            str(feature.feature),
            f"{feature.peak:.6g}",
            f"{feature.mean:.6g}",
            f"{feature.active_fraction:.1%}",
        )
    console.print(table)
    if output is not None:
        console.print(f"Report: [bold]{output}[/bold]")


@app.command()
def causal(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    clean_prompt: Annotated[str, typer.Option("--clean-prompt")],
    corrupt_prompt: Annotated[str, typer.Option("--corrupt-prompt")],
    target_token_id: Annotated[int, typer.Option("--target-token-id", min=0)],
    token_index: Annotated[int, typer.Option("--token", help="Activation token index")] = -1,
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
        token_index=token_index,
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


@app.command("ablate")
def ablate(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    prompt: Annotated[str, typer.Option("--prompt")],
    target_token_id: Annotated[int, typer.Option("--target-token-id", min=0)],
    token_index: Annotated[int, typer.Option("--token", help="Activation token index")] = -1,
    hook: Annotated[
        list[str] | None,
        typer.Option("--hook", help="Glob selecting ablation candidates"),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Rank hook tokens by causal next-token effect under ablation."""

    from ldb.adapters.transformers import TransformersAdapter

    adapter = TransformersAdapter(model, revision=revision, device=device)
    report = causal_ablation(
        adapter,
        prompt=prompt,
        target_token_id=target_token_id,
        token_index=token_index,
        hook_globs=hook or ["block.*.residual.post"],
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    table = Table("Hook", "Ablated log P", "Effect")
    for trial in report.trials:
        table.add_row(
            trial.hook,
            f"{trial.ablated_log_probability:.6g}",
            f"{trial.effect:+.6g}",
        )
    console.print(f"Baseline log P: {report.baseline_log_probability:.6g}")
    console.print(table)
    if output is not None:
        console.print(f"Report: [bold]{output}[/bold]")


@app.command("causal-graph")
def causal_graph(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    clean_prompt: Annotated[str, typer.Option("--clean-prompt")],
    corrupt_prompt: Annotated[str, typer.Option("--corrupt-prompt")],
    target_token_id: Annotated[int, typer.Option("--target-token-id", min=0)],
    token_index: Annotated[int, typer.Option("--token", help="Activation token index")] = -1,
    hook: Annotated[
        list[str] | None,
        typer.Option("--hook", help="Glob selecting causal graph nodes"),
    ] = None,
    minimum_recovery: Annotated[float, typer.Option("--minimum-recovery", min=0.0)] = 0.01,
    max_path_trials: Annotated[int, typer.Option("--max-path-trials", min=1)] = 256,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Build directed causal propagation edges between latent hook points."""

    from ldb.adapters.transformers import TransformersAdapter

    adapter = TransformersAdapter(model, revision=revision, device=device)
    report = causal_propagation_graph(
        adapter,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        target_token_id=target_token_id,
        token_index=token_index,
        hook_globs=hook or ["block.*.residual.post"],
        minimum_recovery=minimum_recovery,
        max_path_trials=max_path_trials,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    table = Table("Source", "Target", "State recovery", "Output effect", "Output recovery")
    for edge in report.edges:
        table.add_row(
            edge.source,
            edge.target,
            f"{edge.activation_recovery:+.2%}",
            f"{edge.mediated_effect:+.6g}",
            f"{edge.mediated_recovery:+.2%}" if edge.mediated_recovery is not None else "-",
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
    token_index: Annotated[
        int | None, typer.Option("--token", help="Scale only this activation token")
    ] = None,
    regression_weight: Annotated[float, typer.Option("--regression-weight", min=0.0)] = 1.0,
    complexity_weight: Annotated[float, typer.Option("--complexity-weight", min=0.0)] = 0.01,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
    patch_output: Annotated[
        Path | None, typer.Option("--patch-output", help="Write the best Neural Patch")
    ] = None,
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
        token_index=token_index,
        regression_weight=regression_weight,
        complexity_weight=complexity_weight,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if patch_output is not None:
        write_patch(patch_output, patch_from_search(report))

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
    if patch_output is not None:
        console.print(f"Neural Patch: [bold]{patch_output}[/bold]")


@app.command("search-combinations")
def search_combinations(
    model: Annotated[str, typer.Argument(help="Hugging Face model identifier")],
    suite: Annotated[Path, typer.Option("--suite", exists=True, dir_okay=False)],
    candidates: Annotated[Path, typer.Option("--candidates", exists=True, dir_okay=False)],
    max_interventions: Annotated[int, typer.Option("--max-interventions", min=1)] = 2,
    max_trials: Annotated[int, typer.Option("--max-trials", min=1)] = 10_000,
    regression_weight: Annotated[float, typer.Option("--regression-weight", min=0.0)] = 1.0,
    complexity_weight: Annotated[float, typer.Option("--complexity-weight", min=0.0)] = 0.01,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
    patch_output: Annotated[
        Path | None, typer.Option("--patch-output", help="Write the best Neural Patch")
    ] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Search bounded combinations of heterogeneous interventions."""

    from ldb.adapters.transformers import TransformersAdapter

    adapter = TransformersAdapter(model, revision=revision, device=device)
    report = search_candidate_combinations(
        adapter,
        load_suite(suite),
        load_candidate_set(candidates),
        resource_directory=candidates.parent,
        max_interventions=max_interventions,
        max_trials=max_trials,
        regression_weight=regression_weight,
        complexity_weight=complexity_weight,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if patch_output is not None:
        patch = patch_from_combination_search(report)
        has_relative_resource = any(
            resource is not None and not Path(resource).is_absolute()
            for intervention in patch.interventions
            for resource in (intervention.tensor, intervention.dictionary)
        )
        if has_relative_resource and patch_output.parent.resolve() != candidates.parent.resolve():
            raise ValueError(
                "A patch with relative resources must be written beside its candidate file"
            )
        write_patch(patch_output, patch)

    table = Table("Candidates", "Target gain", "Related gain", "Regression", "Objective")
    for trial in report.trials[:20]:
        table.add_row(
            ", ".join(trial.candidate_ids) or "<no-op>",
            f"{trial.target_gain:+.6g}",
            f"{trial.related_gain:+.6g}" if trial.related_gain is not None else "-",
            f"{trial.regression_loss:.6g}",
            f"{trial.objective:+.6g}",
        )
    console.print(table)
    if output is not None:
        console.print(f"Report: [bold]{output}[/bold]")
    if patch_output is not None:
        console.print(f"Neural Patch: [bold]{patch_output}[/bold]")


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
    elif name == "hooks" and len(arguments) <= 1:
        method = "hooks"
        if arguments:
            params["pattern"] = arguments[0]
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
        parsed = _parse_patch_arguments(arguments[3:])
        params = {
            "patch_id": patch_id,
            "hook": hook,
            "operation": operation,
            "request_ids": parsed.request_ids,
        }
        for key in (
            "value",
            "tensor",
            "expert",
            "head",
            "num_heads",
            "dictionary",
            "feature",
            "token",
        ):
            if (argument := getattr(parsed, key)) is not None:
                params[key] = argument
        method = "patch.set"
    elif name == "patch-load" and arguments:
        _load_live_patch(client, arguments)
        return True
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
    elif name == "inspect-hit" and arguments:
        if len(arguments) > 2 or (len(arguments) == 2 and arguments[1] != "--capture"):
            raise ValueError("Usage: inspect-hit HIT_ID [--capture]")
        method = "break.inspect"
        params = {"hit_id": arguments[0], "capture": "--capture" in arguments[1:]}
    elif name == "continue" and len(arguments) <= 1:
        method = "continue"
        if arguments:
            params["hit_id"] = arguments[0]
    else:
        raise ValueError(f"Invalid attach command: {command!r}. Type 'help' for usage.")

    results = client.call(method, params)
    console.print_json(data=results)
    return True


def _load_live_patch(client: AgentGroupClient, arguments: list[str]) -> None:
    path = Path(arguments[0]).expanduser()
    request_ids = _parse_request_options(arguments[1:])
    patch = load_patch(path)
    descriptors = client.call("hello")
    mismatched = sorted(
        {str(descriptor["model_id"]) for descriptor in descriptors} - {patch.model_id}
    )
    if mismatched:
        raise ValueError(
            f"Patch targets {patch.model_id!r}, but agents serve {', '.join(mismatched)}"
        )
    if any(
        descriptor.get("model_revision") is None or descriptor.get("tokenizer_hash") is None
        for descriptor in descriptors
    ):
        raise ValueError(
            "Versioned patch loading requires model revision and tokenizer hash from every agent"
        )
    revisions = {str(descriptor["model_revision"]) for descriptor in descriptors}
    tokenizer_hashes = {str(descriptor["tokenizer_hash"]) for descriptor in descriptors}
    if revisions != {patch.model_revision}:
        raise ValueError(
            f"Patch revision {patch.model_revision!r} does not match agents: {sorted(revisions)}"
        )
    if tokenizer_hashes != {patch.tokenizer_hash}:
        raise ValueError("Patch tokenizer hash does not match the attached agents")
    hook_schemas = {str(descriptor["hook_schema_version"]) for descriptor in descriptors}
    if hook_schemas != {patch.hook_schema_version}:
        raise ValueError(
            f"Patch hook schema {patch.hook_schema_version!r} does not match agents: "
            f"{sorted(hook_schemas)}"
        )

    installed: list[str] = []
    try:
        for index, intervention in enumerate(patch.interventions):
            if intervention.request_scoped and not request_ids:
                raise ValueError(
                    f"Intervention {index} is request-scoped; provide --request REQUEST_ID"
                )
            patch_id = f"{patch.patch_id}:{index}"
            params: dict[str, Any] = {
                "patch_id": patch_id,
                "hook": intervention.hook,
                "operation": intervention.operation,
                "request_ids": request_ids if intervention.request_scoped else [],
            }
            if intervention.value is not None:
                params["value"] = intervention.value
            if intervention.tensor is not None:
                tensor_path = Path(intervention.tensor)
                params["tensor"] = str(
                    tensor_path if tensor_path.is_absolute() else path.parent / tensor_path
                )
            if intervention.expert is not None:
                params["expert"] = intervention.expert
            if intervention.head is not None:
                params["head"] = intervention.head
            if intervention.num_heads is not None:
                params["num_heads"] = intervention.num_heads
            if intervention.dictionary is not None:
                dictionary_path = Path(intervention.dictionary)
                params["dictionary"] = str(
                    dictionary_path
                    if dictionary_path.is_absolute()
                    else path.parent / dictionary_path
                )
            if intervention.feature is not None:
                params["feature"] = intervention.feature
            if intervention.token is not None:
                params["token"] = intervention.token
            client.call("patch.set", params)
            installed.append(patch_id)
    except Exception as exc:
        rollback_errors: list[str] = []
        for patch_id in reversed(installed):
            try:
                client.call("patch.remove", {"patch_id": patch_id})
            except Exception as rollback_exc:
                rollback_errors.append(f"{patch_id}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                f"Patch load failed: {exc}; rollback failed: {'; '.join(rollback_errors)}"
            ) from exc
        raise
    console.print(f"Installed {len(installed)} intervention(s) from [bold]{path}[/bold]")


def _parse_request_options(arguments: list[str]) -> list[str]:
    request_ids: list[str] = []
    index = 0
    while index < len(arguments):
        if arguments[index] != "--request" or index + 1 >= len(arguments):
            raise ValueError(f"Invalid request option: {arguments[index]}")
        request_ids.append(arguments[index + 1])
        index += 2
    return request_ids


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


def _parse_patch_arguments(arguments: list[str]) -> _PatchArguments:
    parsed = _PatchArguments()
    index = 0
    if arguments and not arguments[0].startswith("--"):
        parsed.value = float(arguments[0])
        index = 1
    while index < len(arguments):
        option = arguments[index]
        option_fields = {
            "--tensor": "tensor",
            "--expert": "expert",
            "--head": "head",
            "--heads": "num_heads",
            "--dictionary": "dictionary",
            "--feature": "feature",
            "--token": "token",
        }
        if option not in {"--request", *option_fields} or index + 1 >= len(arguments):
            raise ValueError(f"Invalid patch option: {option}")
        index += 1
        if option == "--request":
            parsed.request_ids.append(arguments[index])
        else:
            field_name = option_fields[option]
            if getattr(parsed, field_name) is not None:
                raise ValueError(f"{option} may only be specified once")
            raw = arguments[index]
            setattr(
                parsed,
                field_name,
                raw if option in {"--tensor", "--dictionary"} else int(raw),
            )
        index += 1
    return parsed


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


@patch_app.command("verify")
def verify_neural_patch(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    suite: Annotated[Path, typer.Option("--suite", exists=True, dir_okay=False)],
    target_min_gain: Annotated[float, typer.Option("--target-min-gain")] = 0.0,
    related_min_gain: Annotated[float, typer.Option("--related-min-gain")] = 0.0,
    regression_max_drop: Annotated[float, typer.Option("--regression-max-drop", min=0.0)] = 0.05,
    output: Annotated[Path | None, typer.Option("--output", help="Write JSON report")] = None,
    verified_patch_output: Annotated[
        Path | None,
        typer.Option("--verified-patch-output", help="Write patch with validation results"),
    ] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Verify a Neural Patch against target and regression cases."""

    from ldb.adapters.transformers import TransformersAdapter

    patch = load_patch(path)
    adapter = TransformersAdapter(patch.model_id, revision=patch.model_revision, device=device)
    report = verify_patch(
        adapter,
        patch,
        load_suite(suite),
        patch_directory=path.parent,
        target_min_gain=target_min_gain,
        related_min_gain=related_min_gain,
        regression_max_drop=regression_max_drop,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if report.passed and verified_patch_output is not None:
        verified = patch.model_copy(
            update={
                "validation": ValidationSummary(
                    target_success=report.target_success,
                    related_generalization=report.related_generalization,
                    unrelated_regression=report.unrelated_regression,
                )
            }
        )
        write_patch(verified_patch_output, verified)
    color = "green" if report.passed else "red"
    related = (
        f" related={report.related_generalization:.1%}"
        if report.related_generalization is not None
        else ""
    )
    console.print(
        f"[{color}]{'PASS' if report.passed else 'FAIL'}[/{color}] "
        f"target={report.target_success:.1%}{related} "
        f"worst_regression={report.unrelated_regression:+.6g}"
    )
    table = Table("Category", "Cases", "Success", "Mean delta", "Worst delta")
    for category, summary in report.categories.items():
        table.add_row(
            category,
            str(summary.case_count),
            f"{summary.success:.1%}",
            f"{summary.mean_delta:+.6g}",
            f"{summary.worst_delta:+.6g}",
        )
    console.print(table)
    if output is not None:
        console.print(f"Report: [bold]{output}[/bold]")
    if report.passed and verified_patch_output is not None:
        console.print(f"Verified Neural Patch: [bold]{verified_patch_output}[/bold]")
    if not report.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
