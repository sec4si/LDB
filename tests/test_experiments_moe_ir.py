"""IR correctness, integrity boundaries and executable lowering."""

from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file

from ldb.experiments.moe_ir.core import (
    BOUNDARY,
    VERSION,
    compatibility,
    digest,
    load,
    save,
    swiglu,
    validate,
)
from ldb.experiments.moe_ir.runtime import Runtime


@pytest.fixture
def ir(tmp_path):
    torch.manual_seed(37)
    path = tmp_path / "weights.safetensors"
    tensors = {name: torch.randn(*shape) for name, shape in swiglu(8, 12)["parameters"].items()}
    save_file(tensors, str(path))
    return {
        "schema": VERSION,
        "model": {"name": "fixture"},
        "boundary": dict(BOUNDARY),
        "sources": {"weights": {"uri": path.name, "sha256": digest(path)}},
        "graphs": {"body": swiglu(8, 12)},
        "layers": [
            {
                "id": 0,
                "graph": "body",
                "routing": {
                    "experts": 1,
                    "top_k": 1,
                    "shared_experts": 0,
                    "weighting": "host_defined",
                },
                "experts": [
                    {
                        "id": 0,
                        "bindings": {
                            name: {"source": "weights", "key": name, "storage_dtype": "F32"}
                            for name in tensors
                        },
                    }
                ],
            }
        ],
    }


def test_roundtrip_and_runtime_matches_native(ir, tmp_path):
    save(tmp_path / "ir.json", ir)
    restored = load(tmp_path / "ir.json")
    runtime = Runtime(restored, tmp_path)
    x = torch.randn(11, 8)
    w = runtime.weights
    expected = (torch.nn.functional.silu(x @ w[0, 0, "gate"].T) * (x @ w[0, 0, "up"].T)) @ w[
        0, 0, "down"
    ].T
    torch.testing.assert_close(runtime.expert(0, 0)(x), expected)
    # Mutating caller metadata cannot change the already validated executable.
    restored["graphs"]["body"]["nodes"][0]["op"] = "unknown"
    torch.testing.assert_close(runtime.expert(0, 0)(x), expected)


@pytest.mark.parametrize(
    "fault",
    ["forward", "dimension", "operation", "extra_semantics", "duplicate", "version", "binding"],
)
def test_invalid_ir_rejected_before_execution(ir, fault):
    g = ir["graphs"]["body"]
    if fault == "forward":
        g["nodes"][0]["inputs"] = ["y"]
    elif fault == "dimension":
        g["parameters"]["down"][1] = 99
    elif fault == "operation":
        g["nodes"][2]["op"] = "gelu"
    elif fault == "extra_semantics":
        g["nodes"][0]["bias"] = "ignored-bias"
    elif fault == "duplicate":
        ir["layers"][0]["experts"].append(deepcopy(ir["layers"][0]["experts"][0]))
    elif fault == "version":
        ir["schema"] = "moe.ir/v99"
    else:
        del ir["layers"][0]["experts"][0]["bindings"]["gate"]
    with pytest.raises(ValueError):
        validate(ir)


def test_source_integrity_and_relocation(ir, tmp_path):
    actual = tmp_path / "weights.safetensors"
    ir["sources"]["weights"]["uri"] = "/missing/on/this/machine"
    Runtime(ir, source_overrides={"weights": actual})
    actual.write_bytes(actual.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        Runtime(ir, source_overrides={"weights": actual})


def test_structural_compatibility_is_not_functional(ir):
    other = deepcopy(ir)
    other["layers"][0]["routing"]["experts"] = 2
    second = deepcopy(other["layers"][0]["experts"][0])
    second["id"] = 1
    other["layers"][0]["experts"].append(second)
    result = compatibility(ir, 0, other, 0)
    assert result["expert_structure_compatible"]
    assert not result["whole_layer_structure_compatible"]
    assert result["functional_compatibility"] == "unverified"


def test_input_and_tensor_contracts(ir, tmp_path):
    runtime = Runtime(ir, tmp_path)
    with pytest.raises(ValueError, match="input"):
        runtime.expert(0, 0)(torch.randn(2, 9))
    ir["layers"][0]["experts"][0]["bindings"]["gate"]["storage_dtype"] = "BF16"
    with pytest.raises(ValueError, match="dtype"):
        Runtime(ir, tmp_path)


@pytest.mark.parametrize("fault", [None, "boundary", "identity", "payload"])
def test_existing_package_link_contract(ir, tmp_path, fault):
    import json

    from safetensors.torch import load_file

    from ldb.experiments.moe_ir.bridge import link_package

    ir["model"]["config_sha256"] = "a" * 64
    w = load_file(str(tmp_path / "weights.safetensors"))
    save_file(
        {"0.gate_up": torch.cat([w["gate"], w["up"]]), "0.down": w["down"]},
        str(tmp_path / "bodies.safetensors"),
    )
    save_file({"placeholder": torch.zeros(1)}, str(tmp_path / "interface.safetensors"))
    header = {
        "schema": "moe.expert-module/v0.1",
        "operator": "down(silu(gate(x))*up(x))",
        "dtype": "float32",
        "input": "post-RMSNorm token states before original recipient router dispatch",
        "output": (
            "unweighted expert update before routing-weight multiplication and residual addition"
        ),
        "recipient_checkpoint_sha256": ir["sources"]["weights"]["sha256"],
        "recipient_config_sha256": ir["model"]["config_sha256"],
        "donor_checkpoint_sha256": ir["sources"]["weights"]["sha256"],
        "files": {
            name: digest(tmp_path / name)
            for name in ["bodies.safetensors", "interface.safetensors"]
        },
        "slots": [
            {
                "slot": 0,
                "recipient_layer": 0,
                "recipient_expert": 0,
                "donor_layer": 0,
                "donor_expert": 0,
                "input_dim": 8,
                "output_dim": 8,
            }
        ],
    }
    if fault == "boundary":
        header["output"] = "already_router_weighted"
    elif fault == "identity":
        header["donor_checkpoint_sha256"] = "0" * 64
    elif fault == "payload":
        (tmp_path / "bodies.safetensors").write_bytes(b"changed")
    (tmp_path / "module.json").write_text(json.dumps(header))
    if fault:
        with pytest.raises(ValueError):
            link_package(tmp_path, ir, ir)
    else:
        result = link_package(tmp_path, ir, ir)
        assert len(result["slots"]) == 1 and result["functional_compatibility"] == "unverified"
        assert result["adapter"]["execution"] == "existing expert_abi.package loader"
