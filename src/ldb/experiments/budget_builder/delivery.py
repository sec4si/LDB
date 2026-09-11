"""Seal immutable model assets atomically, excluding interpreter caches."""

import shutil
import tempfile
from pathlib import Path

from .common import read, sha, write


def export_delivery(source, destination, host, spec, candidate, verdict):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not verdict.get("pass"):
        raise ValueError("Cannot package a rejected candidate")
    if destination.exists() or destination.is_relative_to(source):
        raise FileExistsError("Delivery destination must be new and outside the source")
    expected = (
        host["weight_sha256"]
        if candidate["operation"] == "original"
        else read(source / "build-provenance.json")["weight_sha256"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=destination.parent))
    try:
        shutil.copytree(
            source,
            staging,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "budget-model.json"),
        )
        for name, wanted in host.get("tokenizer_sha256", {}).items():
            shutil.copy2(Path(host["tokenizer"]) / name, staging / name)
            if sha(staging / name) != wanted:
                raise ValueError("Copied tokenizer integrity failure")
        weights = {p.name: sha(p) for p in staging.glob("*.safetensors")}
        if weights != expected:
            raise ValueError("Copied weights differ from the built candidate")
        if sum(p.stat().st_size for p in staging.glob("*.safetensors")) > spec["weight_bytes"]:
            raise ValueError("Delivery exceeds weight budget")
        for name, target in [
            ("standalone.py", "budget-run.py"),
            ("prompts.py", "budget_prompts.py"),
        ]:
            shutil.copy2(Path(__file__).with_name(name), staging / target)
        contract = {
            "schema": "budget-model/v1",
            "backend": host["backend"],
            "spec": spec,
            "selected_candidate": candidate,
            "confirmation": verdict,
            "scope": "Single-request fixed workload, arithmetic/extraction tasks and short text loss; no universal capability claim",
            "files": {
                str(p.relative_to(staging)): sha(p) for p in staging.rglob("*") if p.is_file()
            },
        }
        write(staging / "budget-model.json", contract)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return contract
