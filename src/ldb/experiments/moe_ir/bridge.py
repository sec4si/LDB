"""Link existing adapted-body packages to explicit IR host/donor slots."""

import json
from pathlib import Path

from .core import BOUNDARY, compatibility, digest, layer_at, require, swiglu, validate


def link_package(directory, recipient, donor):
    """Verify package tensors and return an inspectable plan, not a quality verdict."""
    from safetensors import safe_open

    validate(recipient)
    validate(donor)
    directory = Path(directory)
    header = json.loads((directory / "module.json").read_text())
    require(header["schema"] == "moe.expert-module/v0.1", "Unsupported module format")
    require(
        header["input"] == "post-RMSNorm token states before original recipient router dispatch"
        and header["output"]
        == "unweighted expert update before routing-weight multiplication and residual addition",
        "Unsupported module host boundary",
    )
    require(
        header["operator"] == "down(silu(gate(x))*up(x))" and header["dtype"] == "float32",
        "Unsupported module operator",
    )
    require(
        header["recipient_checkpoint_sha256"] == recipient["sources"]["weights"]["sha256"]
        and header["recipient_config_sha256"] == recipient["model"]["config_sha256"],
        "Recipient identity mismatch",
    )
    require(
        header["donor_checkpoint_sha256"] == donor["sources"]["weights"]["sha256"],
        "Donor identity mismatch",
    )
    require(
        set(header["files"]) == {"bodies.safetensors", "interface.safetensors"},
        "Incomplete package payload",
    )
    for name, expected in header["files"].items():
        require(digest(directory / name) == expected, "Package payload integrity mismatch")
    require(bool(header["slots"]), "Empty module slots")
    slots, seen, body_slots = [], set(), set()
    with safe_open(directory / "bodies.safetensors", framework="pt") as tensors:
        for slot in header["slots"]:
            rl, re = slot["recipient_layer"], slot["recipient_expert"]
            dl, de = slot["donor_layer"], slot["donor_expert"]
            require(
                type(slot["slot"]) is int and slot["slot"] >= 0 and slot["slot"] not in body_slots,
                "Duplicate or invalid body slot",
            )
            body_slots.add(slot["slot"])
            require((rl, re) not in seen, "Duplicate recipient slot")
            seen.add((rl, re))
            host, source = layer_at(recipient, rl), layer_at(donor, dl)
            require(
                re in {e["id"] for e in host["experts"]}
                and de in {e["id"] for e in source["experts"]},
                "Unknown package expert",
            )
            graph = donor["graphs"][source["graph"]]
            width, intermediate = graph["input_width"], graph["parameters"]["gate"][0]
            require(
                graph == swiglu(width, intermediate), "Module requires the canonical SwiGLU graph"
            )
            require(
                slot["input_dim"] == slot["output_dim"] == width, "Module dimensions differ from IR"
            )
            for role, shape in [
                ("gate_up", [2 * intermediate, width]),
                ("down", [width, intermediate]),
            ]:
                t = tensors.get_slice(f"{slot['slot']}.{role}")
                require(
                    t.get_shape() == shape and t.get_dtype() == "F32",
                    "Packed body differs from IR layout",
                )
            check = compatibility(recipient, rl, donor, dl)
            require(check["expert_structure_compatible"], "IR expert contract mismatch")
            slots.append(
                {
                    "recipient": [rl, re],
                    "donor": [dl, de],
                    "graph": source["graph"],
                    "package_body_slot": slot["slot"],
                    "compatibility": check,
                }
            )
    return {
        "schema": "moe.ir-link/v1",
        "package_header_sha256": digest(directory / "module.json"),
        "boundary": dict(BOUNDARY),
        "slots": slots,
        "adapter": {
            "source": "interface.safetensors",
            "execution": "existing expert_abi.package loader",
            "scope": "H/T remains a backend adapter; not lowered into the v1 expert graph",
        },
        "functional_compatibility": "unverified",
    }
