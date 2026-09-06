from pathlib import Path

import pytest

from ldb.models import HookPhase, HookPoint, RunManifest
from ldb.trace import RunBundleRecorder, load_events, load_manifest

torch = pytest.importorskip("torch")


def test_run_bundle_round_trip(tmp_path: Path) -> None:
    with RunBundleRecorder(tmp_path, run_id="test-run") as recorder:
        recorder.record(
            HookPoint(name="block.0.residual.post", module_path="block", phase=HookPhase.POST),
            torch.tensor([[1.0, 2.0]]),
            capture=True,
        )
        recorder.write_manifest(
            RunManifest(
                run_id="test-run",
                model_id="test-model",
                tokenizer_id="test-model",
                backend="test",
                backend_version="1",
                adapter="test",
                adapter_version="1",
                prompt="hello",
                prompt_token_ids=(1,),
                generation={"max_new_tokens": 1},
                output_text="hello world",
            )
        )

    manifest = load_manifest(tmp_path / "test-run")
    events = load_events(tmp_path / "test-run")

    assert manifest.output_text == "hello world"
    assert events[0].tensor_file == "tensors/00000000.safetensors"
    assert (tmp_path / "test-run" / events[0].tensor_file).is_file()
