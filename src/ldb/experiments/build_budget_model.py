"""Public entry point: inspect a plan or build from user-supplied inputs."""

import argparse
import shutil
from pathlib import Path

from ldb.experiments.budget_builder.common import freeze, read, write
from ldb.experiments.budget_builder.planner import plan
from ldb.experiments.budget_builder.run import execute


def initialize(root, catalog_path, spec_path, development, confirmation):
    catalog, spec = read(catalog_path), read(spec_path)
    for host in catalog["hosts"]:
        if host["backend"] not in ["mlx", "torch"]:
            raise ValueError("Unsupported backend")
        if not host["id"] or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in host["id"]
        ):
            raise ValueError("Host IDs must be simple names")
        for key in ["path", "python", "ir", "tokenizer", "calibration"]:
            if key in host:
                p = Path(host[key])
                host[key] = str((p if p.is_absolute() else catalog_path.parent / p).resolve())
    for swap in catalog.get("swaps", []):
        if not swap["id"] or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in swap["id"]
        ):
            raise ValueError("Candidate IDs must be simple names")
    planned = plan(catalog, spec)
    data = [read(p) for p in [development, confirmation]]
    texts, prompts = set(), set()
    for value in data:
        current = set(value["texts"])
        questions = {q["prompt"] for q in value["tasks"]}
        if not current or not questions or current & texts or questions & prompts:
            raise ValueError("Empty or overlapping evaluation splits")
        if len(current) != len(value["texts"]) or len(questions) != len(value["tasks"]):
            raise ValueError("Duplicate evaluation samples")
        if any(
            q["kind"] not in ["arithmetic", "extract"] or not isinstance(q["answer"], str)
            for q in value["tasks"]
        ):
            raise ValueError("Unsupported task contract")
        texts.update(current)
        prompts.update(questions)
    root.mkdir(parents=True, exist_ok=False)
    for name, value in [
        ("spec.json", spec),
        ("catalog.json", catalog),
        ("plan.json", planned),
        ("development.json", data[0]),
        ("confirmation.json", data[1]),
    ]:
        write(root / name, value)
    package = Path(__file__).parent
    paths = [
        p
        for name in ("budget_builder", "moe_ir", "expert_compaction")
        for p in (package / name).glob("*.py")
    ]
    relative = []
    for path in paths:
        name = Path("ldb/experiments") / path.relative_to(package)
        relative.append(name)
        target = root / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    freeze(
        root,
        ["spec.json", "catalog.json", "plan.json", "development.json", "confirmation.json"]
        + [str(Path("sources") / p) for p in relative],
        "input-freeze.json",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--development", required=True, type=Path)
    p.add_argument("--confirmation", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    initialize(a.output, a.catalog, a.spec, a.development, a.confirmation)
    if not a.plan_only:
        execute(a.output)
    else:
        print("Frozen plan:", a.output / "plan.json")
