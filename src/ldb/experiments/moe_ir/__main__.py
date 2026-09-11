"""python -m ldb.experiments.moe_ir: import, validate, inspect and compare."""

import argparse
import json
from pathlib import Path

from .core import compatibility, load, save

p = argparse.ArgumentParser(description=__doc__)
commands = p.add_subparsers(dest="command", required=True)
imp = commands.add_parser("import-study")
imp.add_argument("--root", type=Path, required=True)
imp.add_argument("--output", type=Path, required=True)
for command in ["validate", "inspect"]:
    sub = commands.add_parser(command)
    sub.add_argument("file", type=Path)
compare = commands.add_parser("compare")
compare.add_argument("left", type=Path)
compare.add_argument("right", type=Path)
compare.add_argument("--left-layer", type=int, required=True)
compare.add_argument("--right-layer", type=int, required=True)
a = p.parse_args()
if a.command == "import-study":
    from .importer import from_study

    a.output.mkdir(parents=True, exist_ok=False)
    records = json.loads((a.root / "models.json").read_text())
    for i in range(len(records)):
        ir = from_study(a.root, i)
        save(a.output / f"model-{i}.json", ir)
        print(f"Imported {ir['model']['name']}: {len(ir['layers'])} MoE layers")
elif a.command == "compare":
    print(
        json.dumps(
            compatibility(load(a.left), a.left_layer, load(a.right), a.right_layer), indent=2
        )
    )
else:
    ir = load(a.file)
    result = {
        "valid": True,
        "model": ir["model"]["name"],
        "layers": len(ir["layers"]),
        "experts": sum(len(layer_row["experts"]) for layer_row in ir["layers"]),
    }
    if a.command == "inspect":
        result["profiles"] = [
            {
                "layer": layer_row["id"],
                "graph": layer_row["graph"],
                "routing": layer_row["routing"],
                "observations": layer_row.get("observations"),
            }
            for layer_row in ir["layers"]
        ]
    print(json.dumps(result, indent=2))
