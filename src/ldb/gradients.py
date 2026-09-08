"""Non-mutating gradient observations for existing eager PyTorch training loops."""

from __future__ import annotations

import fnmatch
import json
import math
import shutil
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


def tensor_leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, f"{path}.{index}")


def tensor_stats(tensor: Any) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "numel": tensor.numel(),
    }
    if tensor.layout != torch.strided:
        result["unsupported_layout"] = True
        return result
    with torch.no_grad():
        detached = tensor.detach()
        nonfinite = int((~torch.isfinite(detached)).sum())
        values = detached.abs()
        if values.dtype != torch.float64:
            values = values.float()
        norm = float(values.norm())
        maximum = float(values.max()) if values.numel() else 0.0
        result.update(
            nonfinite=nonfinite,
            norm=norm if math.isfinite(norm) else None,
            max_abs=maximum if math.isfinite(maximum) else None,
            zero_fraction=float((values == 0).float().mean()) if values.numel() else None,
        )
    return result


def select_names(names: Sequence[str], patterns: Sequence[str]) -> list[str]:
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in names):
            raise ValueError(f"Pattern matches no names: {pattern}")
    return [name for name in names if any(fnmatch.fnmatchcase(name, p) for p in patterns)]


class GradientTracer:
    """Observe incoming and accumulated gradients without returning replacements.

    Use with one exclusive eager training loop. Loss scaling metadata is explicit;
    this tracer does not unscale, clip, zero gradients, or change model mode.
    """

    def __init__(
        self,
        model: Any,
        *,
        parameters: Sequence[str] = ("*",),
        modules: Sequence[str] = (),
        capture: bool = False,
        max_events: int = 2000,
        max_tensor_bytes: int = 64 * 1024 * 1024,
        norm_threshold: float | None = None,
    ) -> None:
        if max_events < 1 or max_tensor_bytes < 0:
            raise ValueError("Invalid trace budget")
        if norm_threshold is not None and (not math.isfinite(norm_threshold) or norm_threshold < 0):
            raise ValueError("norm_threshold must be finite and nonnegative")
        named = dict(model.named_parameters(remove_duplicate=False))
        selected = select_names(list(named), parameters)
        self.parameters: dict[str, Any] = {}
        seen: set[int] = set()
        for name in selected:
            if id(named[name]) not in seen:
                self.parameters[name] = named[name]
                seen.add(id(named[name]))
        named_modules = dict(model.named_modules())
        self.modules = {
            name: named_modules[name] for name in select_names(list(named_modules), modules)
        }
        self.capture = capture
        self.max_events = max_events
        self.max_tensor_bytes = max_tensor_bytes
        self.norm_threshold = norm_threshold
        self.events: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.tensors: dict[str, Any] = {}
        self.tensor_bytes = 0
        self.dropped_events = 0
        self.skipped_tensors = 0
        self.handles: list[Any] = []
        self.activation_handles: list[Any] = []
        self.active = False
        self.started = False
        self.step: str = "unmarked"
        self.loss_scale = 1.0

    def mark_step(self, step: str | int, *, loss_scale: float = 1.0) -> None:
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive")
        self.step, self.loss_scale = str(step), loss_scale

    def record(self, name: str, kind: str, tensor: Any) -> None:
        import torch

        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        event: dict[str, Any] = {
            "index": len(self.events),
            "step": self.step,
            "loss_scale": self.loss_scale,
            "name": name,
            "kind": kind,
            "stats": tensor_stats(tensor),
            "alerts": [],
        }
        stats = event["stats"]
        if stats.get("nonfinite", 0):
            event["alerts"].append("nonfinite")
        elif "norm" in stats and stats["norm"] is None:
            event["alerts"].append("statistic_nonfinite")
        if (
            self.norm_threshold is not None
            and stats.get("norm") is not None
            and stats["norm"] > self.norm_threshold
        ):
            event["alerts"].append("norm_threshold")
        if self.capture:
            size = tensor.numel() * tensor.element_size()
            if tensor.layout == torch.strided and self.tensor_bytes + size <= self.max_tensor_bytes:
                key = f"event_{event['index']}"
                self.tensors[key] = tensor.detach().cpu().contiguous().clone()
                self.tensor_bytes += size
                event["tensor_key"] = key
            else:
                self.skipped_tensors += 1
                event["capture_skipped"] = True
        self.events.append(event)

    def _observe_module(self, name: str, output: Any) -> None:
        for path, tensor in tensor_leaves(output):
            if len(self.activation_handles) >= self.max_events:
                self.dropped_events += 1
                continue
            self.record(name + path, "activation", tensor)
            if tensor.requires_grad:
                handle = tensor.register_hook(
                    lambda grad, key=name + path: self.record(key, "activation_gradient", grad)
                )
                self.activation_handles.append(handle)

    def __enter__(self) -> GradientTracer:
        if self.started:
            raise ValueError("A tracer is single-use; create a new trace for another window")
        self.started = True
        self.active = True
        try:
            for name, parameter in self.parameters.items():
                if not parameter.requires_grad:
                    continue
                self.handles.append(
                    parameter.register_hook(
                        lambda grad, key=name: self.record(key, "incoming_gradient", grad)
                    )
                )
                self.handles.append(
                    parameter.register_post_accumulate_grad_hook(
                        lambda param, key=name: self.record(key, "accumulated_gradient", param.grad)
                    )
                )
            for name, module in self.modules.items():
                self.handles.append(
                    module.register_forward_hook(
                        lambda mod, args, output, key=name: self._observe_module(key, output)
                    )
                )
        except BaseException:
            self.close()
            raise
        return self

    def snapshot(self, label: str = "snapshot") -> dict[str, str]:
        states = {}
        for name, parameter in self.parameters.items():
            states[name] = (
                "frozen"
                if not parameter.requires_grad
                else ("none" if parameter.grad is None else "present")
            )
            if parameter.grad is not None:
                self.record(name, label, parameter.grad)
        if len(self.snapshots) < self.max_events:
            self.snapshots.append({"step": self.step, "label": label, "states": states})
        else:
            self.dropped_events += 1
        return states

    def close(self) -> None:
        for handle in [*self.handles, *self.activation_handles]:
            handle.remove()
        self.handles.clear()
        self.activation_handles.clear()
        self.active = False

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def save(self, directory: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        from safetensors.torch import save_file

        if self.active:
            raise ValueError("Close the trace before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        report = {
            "schema": "ldb.gradient-trace/v1alpha1",
            "metadata": metadata or {},
            "events": self.events,
            "snapshots": self.snapshots,
            "parameters": list(self.parameters),
            "modules": list(self.modules),
            "dropped_events": self.dropped_events,
            "skipped_tensors": self.skipped_tensors,
            "tensor_bytes": self.tensor_bytes,
            "gradient_units": (
                "Observed tensors; loss_scale is declared metadata, no automatic unscaling"
            ),
        }
        try:
            if self.tensors:
                save_file(self.tensors, str(directory / "tensors.safetensors"))
            (directory / "trace.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
        except BaseException:
            shutil.rmtree(directory)
            raise
        return report
