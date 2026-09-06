"""Portable RunBundle recording and inspection."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter
from safetensors.torch import save_file

from ldb.models import HookPoint, RunManifest, TensorSummary, TraceEvent


class RunBundleRecorder:
    """Append trace events and selected tensors to a portable directory."""

    def __init__(self, root: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.path = root / self.run_id
        self.tensor_path = self.path / "tensors"
        self.tensor_path.mkdir(parents=True, exist_ok=False)
        self._events = (self.path / "events.jsonl").open("x", encoding="utf-8")
        self._sequence = 0

    def record(self, hook: HookPoint, tensor: object, *, capture: bool) -> TraceEvent:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor at {hook.name}, got {type(tensor).__name__}")

        value = tensor.detach()
        stats = value.float()
        summary = TensorSummary(
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            device=str(value.device),
            mean=stats.mean().item() if value.numel() else None,
            std=stats.std(correction=0).item() if value.numel() else None,
            norm=stats.norm().item() if value.numel() else None,
        )

        tensor_file = None
        if capture:
            tensor_file = f"tensors/{self._sequence:08d}.safetensors"
            save_file({"activation": value.contiguous().cpu()}, self.path / tensor_file)

        event = TraceEvent(
            sequence=self._sequence,
            timestamp=datetime.now(UTC),
            hook=hook.name,
            phase=hook.phase,
            summary=summary,
            tensor_file=tensor_file,
        )
        self._events.write(event.model_dump_json() + "\n")
        self._events.flush()
        self._sequence += 1
        return event

    def write_manifest(self, manifest: RunManifest) -> None:
        if manifest.run_id != self.run_id:
            raise ValueError("Manifest run_id does not match the RunBundle")
        (self.path / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    def close(self) -> None:
        self._events.close()

    def __enter__(self) -> RunBundleRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_manifest(bundle: Path) -> RunManifest:
    return RunManifest.model_validate_json((bundle / "manifest.json").read_text(encoding="utf-8"))


def load_events(bundle: Path) -> list[TraceEvent]:
    adapter = TypeAdapter(TraceEvent)
    with (bundle / "events.jsonl").open(encoding="utf-8") as stream:
        return [adapter.validate_json(line) for line in stream if line.strip()]


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
