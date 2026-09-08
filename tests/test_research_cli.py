import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from typer.testing import CliRunner

from ldb.cli import app


@pytest.fixture
def local_model(tmp_path):
    folder = tmp_path / "model"
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=2,
            n_embd=8,
            vocab_size=8,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
    ).eval()
    model.save_pretrained(folder)
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "John": 1, "Mary": 2, "gave": 3, "a": 4, "book": 5, "to": 6, "Tom": 7},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[UNK]", pad_token="[UNK]"
    )
    tokenizer.save_pretrained(folder)
    return str(folder)


def test_weight_cli_reload(local_model, tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "weights",
            "export",
            local_model,
            "--output",
            str(tmp_path / "new"),
            "--parameter",
            "lm_head.weight",
            "--operation",
            "scale",
            "--value",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["edit"]["before_hash"] != report["edit"]["after_hash"]
    assert PreTrainedTokenizerFast.from_pretrained(tmp_path / "new").encode("John") == [1]
    listed = CliRunner().invoke(app, ["weights", "list", local_model, "--pattern", "lm_head.*"])
    assert listed.exit_code == 0, listed.output
    assert "lm_head.weight" in json.loads(listed.stdout)


def test_gradient_cli_training_and_validation(local_model, tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "gradients",
            local_model,
            "--text",
            "John gave a book to Mary",
            "--output",
            str(tmp_path / "trace"),
            "--steps",
            "2",
            "--learning-rate",
            "0.01",
            "--parameter",
            "transformer.h.0.*",
            "--module",
            "transformer.h.0.mlp",
            "--capture",
        ],
    )
    assert result.exit_code == 0, result.output
    trace = json.loads((tmp_path / "trace/trace.json").read_text())
    assert trace["metadata"]["optimizer_updates"] == 2
    assert len(trace["metadata"]["losses"]) == 2
    assert all(trace["snapshots"][0]["states"].values())
    assert any(e["kind"] == "activation_gradient" for e in trace["events"])
    invalid = CliRunner().invoke(
        app, ["gradients", local_model, "--text", "John", "--output", str(tmp_path / "invalid")]
    )
    assert invalid.exit_code != 0 and "two tokens" in invalid.output


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_operator_cli_forward_backward_and_early_exit(local_model, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Requires Apple MPS")
    runner = CliRunner()
    for options in ([], ["--backward"]):
        result = runner.invoke(
            app,
            [
                "debug-ops",
                local_model,
                "--prompt",
                "John gave a book",
                "--device",
                device,
                *options,
                "-e",
                "step",
                "-e",
                "continue",
            ],
        )
        assert result.exit_code == 0, str(result.exception) + result.output
        assert '"state": "finished"' in result.output
    early = runner.invoke(app, ["debug-ops", local_model, "--prompt", "John gave", "-e", "inspect"])
    assert early.exit_code == 0, early.output
