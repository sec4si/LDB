from pathlib import Path

import pytest
from pydantic import ValidationError

from ldb.models import Intervention, InterventionKind, NeuralPatch
from ldb.patches import load_patch


def test_example_patch_is_valid() -> None:
    patch = load_patch(Path("examples/patches/scale-feature.yaml"))

    assert patch.patch_id == "ldb-example-0001"
    assert patch.interventions[0].request_scoped is True


def test_patch_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        NeuralPatch.model_validate(
            {
                "patch_id": "bad",
                "description": "invalid",
                "model_id": "model",
                "model_revision": "revision",
                "tokenizer_hash": "hash",
                "adapter": "adapter",
                "interventions": [{"hook": "logits", "operation": "ablate"}],
                "unexpected": True,
            }
        )


def test_feature_scale_requires_dictionary_and_feature() -> None:
    intervention = Intervention(
        hook="block.0.residual.post",
        operation=InterventionKind.FEATURE_SCALE,
        value=0.4,
        dictionary="block-0.safetensors",
        feature=12,
    )
    assert intervention.feature == 12

    with pytest.raises(ValidationError, match="Invalid arguments"):
        Intervention(
            hook="block.0.residual.post",
            operation=InterventionKind.FEATURE_SCALE,
            value=0.4,
        )
