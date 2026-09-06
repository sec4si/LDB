"""Shared offline intervention installation."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from ldb.causal import CausalRuntime
from ldb.interventions import apply_tensor_operation, load_intervention_tensor
from ldb.models import HookPoint, Intervention, InterventionKind
from ldb.runtime import ProbeHandle
from ldb.sae import SAEDictionary


def install_interventions(
    runtime: CausalRuntime,
    interventions: tuple[Intervention, ...],
    *,
    resource_directory: Path,
) -> list[ProbeHandle]:
    handles: list[ProbeHandle] = []
    try:
        for intervention in interventions:
            matches = tuple(
                hook
                for hook in runtime.hook_points()
                if fnmatch.fnmatchcase(hook.name, intervention.hook)
            )
            if not matches:
                raise ValueError(f"Patch hook did not match this runtime: {intervention.hook}")
            resource = _load_resource(intervention, resource_directory)

            def apply(
                _: HookPoint,
                tensor: Any,
                operation: InterventionKind = intervention.operation,
                value: float | None = intervention.value,
                loaded: Any = resource,
                expert: int | None = intervention.expert,
                head: int | None = intervention.head,
                num_heads: int | None = intervention.num_heads,
                token: int | None = intervention.token,
            ) -> Any:
                return apply_tensor_operation(
                    tensor,
                    operation,
                    value=value,
                    replacement=loaded,
                    expert=expert,
                    head=head,
                    num_heads=num_heads,
                    feature=loaded,
                    token=token,
                )

            handles.extend(runtime.install_intervention(hook, apply) for hook in matches)
    except Exception:
        remove_interventions(handles)
        raise
    return handles


def remove_interventions(handles: list[ProbeHandle]) -> None:
    for handle in reversed(handles):
        handle.remove()


def _load_resource(intervention: Intervention, directory: Path) -> Any:
    if intervention.dictionary is not None:
        assert intervention.feature is not None
        return SAEDictionary(_resolve(directory, intervention.dictionary)).feature(
            intervention.feature
        )
    if intervention.tensor is not None:
        return load_intervention_tensor(_resolve(directory, intervention.tensor))
    return None


def _resolve(directory: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else directory / path
