"""Portable, declarative donor-body packages with explicit recipient contracts."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .common import batches, load, read, sha, write
from .interfaces import forward, load_maps, save_maps
from .model import replace, weights


def export(root):
    records = read(root / "models.json")
    packages = []
    for choice in read(root / "choices.json"):
        for side, other in [("a", "b"), ("b", "a")]:
            recipient, donor = choice[side], choice[other]
            directory = root / "packages" / f"{recipient}-from-{donor}"
            directory.mkdir(parents=True, exist_ok=False)
            maps = load_maps(root / "interfaces" / f"{recipient}-from-{donor}-rank32.safetensors")
            save_maps(directory / "interface.safetensors", {k: maps[k] for k in ["H", "T"]})
            payload, slots = {}, []
            for i, pair in enumerate(choice["selected"]):
                dl, de = pair[f"{other}_layer"], pair[f"{other}_expert"]
                gu, down = weights(records[donor], dl, de)
                payload[f"{i}.gate_up"], payload[f"{i}.down"] = gu.contiguous(), down.contiguous()
                slots.append(
                    dict(
                        slot=i,
                        recipient_layer=pair[f"{side}_layer"],
                        recipient_expert=pair[f"{side}_expert"],
                        donor_layer=dl,
                        donor_expert=de,
                        input_dim=gu.shape[1],
                        output_dim=down.shape[0],
                        body_parameters=gu.numel() + down.numel(),
                    )
                )
            save_file(payload, str(directory / "bodies.safetensors"))
            manifest = dict(
                schema="moe.expert-module/v0.1",
                operator="down(silu(gate(x))*up(x))",
                input="post-RMSNorm token states before original recipient router dispatch",
                output="unweighted expert update before routing-weight multiplication and residual addition",
                dtype="float32",
                recipient=recipient,
                donor=donor,
                recipient_checkpoint_sha256=records[recipient]["weight_sha256"],
                recipient_config_sha256=records[recipient]["config_sha256"],
                donor_checkpoint_sha256=records[donor]["weight_sha256"],
                slots=slots,
                interface_reuse="One H/T pair for all listed slots",
                files={p.name: sha(p) for p in directory.glob("*.safetensors")},
                compatibility="Only declared target checkpoint and slots have been tested; no semantic or general portability guarantee",
            )
            write(directory / "module.json", manifest)
            packages.append(str(directory.relative_to(root)))
    write(root / "package-index.json", packages)


def load_package(directory, record):
    manifest = read(directory / "module.json")
    if manifest["schema"] != "moe.expert-module/v0.1" or manifest["dtype"] != "float32":
        raise ValueError("Unsupported package ABI")
    if (
        manifest["recipient_checkpoint_sha256"] != record["weight_sha256"]
        or manifest["recipient_config_sha256"] != record["config_sha256"]
    ):
        raise ValueError("Recipient identity mismatch")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError("Package artifact integrity failure")
    maps = load_maps(directory / "interface.safetensors")
    tensors = load_file(str(directory / "bodies.safetensors"))
    replacements = []
    for slot in manifest["slots"]:
        i = slot["slot"]
        payload = tensors[f"{i}.gate_up"], tensors[f"{i}.down"]
        d, out = slot["input_dim"], slot["output_dim"]
        if payload[0].shape != (2 * payload[1].shape[1], d) or payload[1].shape[0] != out:
            raise ValueError("Body dimensions violate package contract")

        def function(x, payload=payload):
            if x.dtype != torch.float32 or x.shape[-1] != payload[0].shape[1]:
                raise ValueError("Input violates package dtype/dimension contract")
            return forward(x, payload, maps)

        replacements.append((slot["recipient_layer"], slot["recipient_expert"], function))
    return manifest, replacements


def verify_packages(root):
    torch.set_num_threads(4)
    records = read(root / "models.json")
    results = []
    for name in read(root / "package-index.json"):
        directory = root / name
        header = read(directory / "module.json")
        recipient, donor = header["recipient"], header["donor"]
        manifest, restored = load_package(directory, records[recipient])
        maps = load_maps(root / "interfaces" / f"{recipient}-from-{donor}-rank32.safetensors")
        direct = []
        for slot, (_, _, function) in zip(manifest["slots"], restored, strict=True):
            payload = weights(records[donor], slot["donor_layer"], slot["donor_expert"])
            with np.load(
                root
                / "captures"
                / f"model-{recipient}"
                / "heldout"
                / f"layer-{slot['recipient_layer']}.npz"
            ) as data:
                x = torch.from_numpy(data["x"].copy())
            if not torch.equal(function(x), forward(x, payload, maps)):
                raise RuntimeError("Package body output differs after reload")
            direct.append(
                (
                    slot["recipient_layer"],
                    slot["recipient_expert"],
                    lambda x, p=payload: forward(x, p, maps),
                )
            )
        model = load(records[recipient])
        batch = batches(root, recipient, "heldout")[0]
        with torch.inference_mode():
            original = model(**batch, use_cache=False, output_router_logits=False).logits
            with replace(model, direct):
                expected = model(**batch, use_cache=False, output_router_logits=False).logits
            with replace(model, restored):
                actual = model(**batch, use_cache=False, output_router_logits=False).logits
            if not torch.equal(actual, expected):
                raise RuntimeError("Reloaded package changes full-model logits")
            if not torch.equal(
                original, model(**batch, use_cache=False, output_router_logits=False).logits
            ):
                raise RuntimeError("Package rollback failed")
        results.append(
            dict(
                package=name,
                header_sha256=sha(directory / "module.json"),
                sampled_states_per_slot=len(x),
                body_outputs_bitwise_equal=True,
                full_model_batch_logits_bitwise_equal=True,
                rollback_bitwise_equal=True,
            )
        )
        print("Package verified", name, flush=True)
        del model
    write(
        root / "package-verification.json",
        dict(packages=results, source_sha256=sha(Path(__file__))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export", "verify"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    (export if args.stage == "export" else verify_packages)(args.root)
