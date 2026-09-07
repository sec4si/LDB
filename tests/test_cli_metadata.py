from pathlib import Path

from typer.testing import CliRunner

from ldb.cli import _portable_diagnosis_patch, app
from ldb.models import Intervention, InterventionKind, NeuralPatch

runner = CliRunner()


def test_runtimes_lists_audited_serving_versions() -> None:
    result = runner.invoke(app, ["runtimes"])

    assert result.exit_code == 0
    assert '"sglang": [' in result.stdout
    assert '"0.5.19"' in result.stdout
    assert '"vllm": [' in result.stdout
    assert '"0.28.0"' in result.stdout


def test_diagnosis_patch_packages_sae_dictionary(tmp_path: Path) -> None:
    dictionary = tmp_path / "source.safetensors"
    dictionary.write_bytes(b"dictionary")
    patch = NeuralPatch(
        patch_id="patch",
        description="feature patch",
        model_id="model",
        model_revision="revision",
        tokenizer_hash="tokenizer",
        adapter="adapter",
        interventions=(
            Intervention(
                hook="block.0.residual.post",
                operation=InterventionKind.FEATURE_SCALE,
                value=0.5,
                dictionary=str(dictionary),
                feature=0,
            ),
        ),
    )

    portable = _portable_diagnosis_patch(tmp_path / "patch.yaml", patch)

    resource = Path(portable.interventions[0].dictionary or "")
    assert not resource.is_absolute()
    assert (tmp_path / resource).read_bytes() == b"dictionary"
