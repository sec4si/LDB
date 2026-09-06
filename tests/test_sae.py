from pathlib import Path

import pytest
from safetensors.torch import save_file

from ldb.sae import SAEDictionary, inspect_sae_features

torch = pytest.importorskip("torch")


def test_sae_inspector_ranks_features_by_peak(tmp_path: Path) -> None:
    path = tmp_path / "dictionary.safetensors"
    save_file(
        {
            "encoder": torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
            "decoder": torch.eye(2),
            "encoder_bias": torch.zeros(2),
            "decoder_bias": torch.zeros(2),
        },
        path,
    )
    dictionary = SAEDictionary(path)
    features = inspect_sae_features(torch.tensor([[[1.0, 1.0], [3.0, -1.0]]]), dictionary, top_k=2)

    assert dictionary.model_width == 2
    assert dictionary.feature_count == 2
    assert [feature.feature for feature in features] == [0, 1]
    assert features[0].peak == 3.0
    assert features[1].active_fraction == 0.5
