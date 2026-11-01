from pathlib import Path

import pytest

from ldb.crosscoder import diff_crosscoder

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file


def test_crosscoder_diff_separates_exclusive_shared_and_dead_features(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    decoder = torch.zeros(7, 2, 4)
    decoder[0:2, 0] = torch.randn(2, 4, generator=generator)
    decoder[2:4, 1] = torch.randn(2, 4, generator=generator)
    shared = torch.randn(2, 4, generator=generator)
    decoder[4:6, 0] = shared
    decoder[4:6, 1] = -shared
    save_file({"decoder": decoder}, tmp_path / "crosscoder.safetensors")

    report = diff_crosscoder(tmp_path / "crosscoder.safetensors", model_a="base", model_b="chat")

    assert (report.a_only, report.shared, report.b_only, report.dead_features) == (2, 2, 2, 1)
    assert report.shared_cosine_mean == pytest.approx(-1.0)
    assert report.relative_norm_histogram == (2, 0, 0, 0, 0, 2, 0, 0, 0, 2)
    assert {feature.feature for feature in report.top_a_only} == {0, 1}
    assert {feature.feature for feature in report.top_b_only} == {2, 3}


def test_crosscoder_diff_requires_a_two_model_decoder(tmp_path: Path) -> None:
    save_file({"decoder": torch.zeros(3, 4)}, tmp_path / "flat.safetensors")
    save_file({"encoder": torch.zeros(2, 4, 3)}, tmp_path / "none.safetensors")

    with pytest.raises(ValueError, match="shape"):
        diff_crosscoder(tmp_path / "flat.safetensors")
    with pytest.raises(ValueError, match="missing"):
        diff_crosscoder(tmp_path / "none.safetensors")
