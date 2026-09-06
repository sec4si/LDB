"""Neural Patch serialization and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ldb.models import NeuralPatch


def load_patch(path: Path) -> NeuralPatch:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return NeuralPatch.model_validate(raw)


def write_patch_schema(path: Path) -> None:
    path.write_text(json.dumps(NeuralPatch.model_json_schema(), indent=2) + "\n", encoding="utf-8")
