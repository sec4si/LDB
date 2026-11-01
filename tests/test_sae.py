from pathlib import Path

import pytest
from safetensors.torch import save_file

from ldb.sae import SAEDictionary, inspect_sae_features, inspect_sae_tokens, transcoder_error

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
    assert features[0].peak_token == 1
    assert features[1].active_fraction == 0.5

    tokens = inspect_sae_tokens(torch.tensor([[[1.0, 1.0], [3.0, -1.0]]]), dictionary, top_k=1)
    assert tokens[0].features[0].feature == 1
    assert tokens[0].features[0].value == 2.0
    assert tokens[1].features[0].feature == 0
    assert tokens[1].features[0].value == 3.0


def test_dictionary_centers_on_decoder_bias_unless_an_input_bias_is_given(tmp_path: Path) -> None:
    base = {"encoder": torch.eye(2), "decoder": torch.eye(2), "decoder_bias": torch.ones(2)}
    save_file(base, tmp_path / "sae.safetensors")
    save_file({**base, "input_bias": torch.zeros(2)}, tmp_path / "transcoder.safetensors")
    activation = torch.tensor([[3.0, 3.0]])

    assert SAEDictionary(tmp_path / "sae.safetensors").encode(activation).tolist() == [[2.0, 2.0]]
    transcoder = SAEDictionary(tmp_path / "transcoder.safetensors")
    assert transcoder.encode(activation).tolist() == [[3.0, 3.0]]
    assert transcoder.decode(transcoder.encode(activation)).tolist() == [[4.0, 4.0]]


def test_transcoder_error_measures_reconstruction_of_the_real_mlp_output(tmp_path: Path) -> None:
    weight = torch.tensor([[2.0, 0.0], [1.0, -1.0]])
    bias = torch.tensor([0.5, -0.5])
    save_file(
        {
            "encoder": torch.cat([torch.eye(2), -torch.eye(2)], 1),
            "decoder": torch.cat([weight, -weight], 0),
            "decoder_bias": bias,
            "input_bias": torch.zeros(2),
        },
        tmp_path / "exact.safetensors",
    )
    save_file(
        {"encoder": torch.zeros(2, 4), "decoder": torch.zeros(4, 2)}, tmp_path / "zero.safetensors"
    )
    inputs = torch.tensor([[[1.0, -2.0], [0.5, 3.0], [-1.0, 1.0]]])
    outputs = inputs @ weight + bias
    target = outputs.reshape(-1, 2)
    expected = target.square().sum() / (target - target.mean(0)).square().sum()

    assert transcoder_error(inputs, outputs, SAEDictionary(tmp_path / "exact.safetensors")) == (
        pytest.approx(0.0, abs=1e-10)
    )
    assert transcoder_error(inputs, outputs, SAEDictionary(tmp_path / "zero.safetensors")) == (
        pytest.approx(float(expected))
    )
