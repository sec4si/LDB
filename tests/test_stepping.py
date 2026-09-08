from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ldb.stepping import LayerDebugger


def test_steps_match_normal_forward_and_snapshot_replay() -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    with torch.inference_mode():
        expected = model(**inputs, use_cache=False).logits
    debugger = LayerDebugger(model, inputs)
    calls = []
    handle = model.transformer.h[0].register_forward_hook(lambda *args: calls.append(0))
    debugger.step()
    debugger.snapshot("middle")
    debugger.step()
    assert calls == [0]
    assert torch.equal(debugger.logits(), expected)
    debugger.restore("middle")
    debugger.step()
    assert calls == [0]
    assert torch.equal(debugger.logits(), expected)
    handle.remove()


def test_conditions_branch_and_patch_history() -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    debugger = LayerDebugger(model, {"input_ids": torch.tensor([[1, 2]])})
    debugger.command("break 0 norm 0")
    assert debugger.run()["stop_reason"] == "breakpoint"
    saved = debugger.hidden.clone()
    assert debugger.branch(1, 0)["max_logit_delta"] > 0
    assert debugger.index == 1 and torch.equal(debugger.hidden, saved)
    debugger.command("patch 1 0")
    debugger.command("undo")
    assert debugger.patches == {}
    debugger.command("redo")
    assert debugger.patches == {1: 0}
    debugger.command("watch norm 0")
    assert debugger.run()["stop_reason"] == "watch"
    with pytest.raises(ValueError, match="executed"):
        debugger.command("undo")
    debugger.restore("start")
    assert debugger.patches == {} and debugger.index == 0


def test_disk_checkpoint_replays_in_new_session_and_rejects_wrong_inputs(tmp_path: Path) -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    inputs = {"input_ids": torch.tensor([[1, 2]])}
    debugger = LayerDebugger(model, inputs)
    debugger.step()
    debugger.patch(1, 0.5)
    path = tmp_path / "middle.safetensors"
    debugger.save(path)
    with pytest.raises(FileExistsError):
        debugger.save(path)
    debugger.run()
    expected = debugger.logits().clone()
    restored = LayerDebugger(model, inputs)
    restored.load(path)
    assert restored.index == 1 and restored.patches == {1: 0.5}
    restored.run()
    assert torch.equal(restored.logits(), expected)
    wrong = LayerDebugger(model, {"input_ids": torch.tensor([[1, 3]])})
    before = wrong.hidden.clone()
    with pytest.raises(ValueError, match="do not match"):
        wrong.load(path)
    assert wrong.index == 0 and torch.equal(wrong.hidden, before)
    with torch.no_grad():
        model.transformer.h[0].mlp.c_proj.weight.add_(1)
    with pytest.raises(ValueError, match="do not match"):
        restored.load(path)


def test_sae_watch_is_layer_specific_and_reports_feature(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    dictionary = tmp_path / "sae.safetensors"
    save_file(
        {"encoder": torch.zeros(8, 1), "decoder": torch.zeros(1, 8), "encoder_bias": torch.ones(1)},
        dictionary,
    )
    debugger = LayerDebugger(model, {"input_ids": torch.tensor([[1, 2]])})
    debugger.watch_feature("feature-one", 1, dictionary, 0, 0.5)
    assert debugger.step()["features"] == {}
    state = debugger.run()
    assert state["stop_reason"] == "watch"
    assert state["watch_hits"] == ["feature-one"]
    assert state["features"]["feature-one"]["peak"] == 1.0
    debugger.command("unwatch-feature feature-one")
    assert debugger.feature_watches == {}


def test_head_token_edit_matches_direct_hook_and_compare_restores_history() -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

    def ablate_last_head(_, args):
        value = args[0].clone()
        value[:, -1, 4:] = 0
        return (value, *args[1:])

    handle = model.transformer.h[1].attn.c_proj.register_forward_pre_hook(ablate_last_head)
    with torch.inference_mode():
        expected = model(**inputs, use_cache=False).logits
    handle.remove()
    debugger = LayerDebugger(model, inputs)
    debugger.command("target-ids 3 4")
    debugger.command("patch-token first 1 0 1")
    debugger.command("undo")
    history = list(debugger.redo_stack)
    hidden = debugger.hidden.clone()
    comparison = debugger.command('compare "patch-head last 1 1 0 -1"')
    assert debugger.index == 0 and debugger.edits == {}
    assert torch.equal(debugger.hidden, hidden) and debugger.redo_stack == history
    assert comparison["patched"]["margin"][0] == pytest.approx(
        float(expected[0, -1, 3] - expected[0, -1, 4]), abs=1e-6
    )
    debugger.command("patch-head last 1 1 0 -1")
    debugger.run()
    assert torch.equal(debugger.logits(), expected)
    assert not model.transformer.h[1].attn.c_proj._forward_pre_hooks
    with pytest.raises(ValueError, match="executed"):
        debugger.command("unpatch-edit last")


def test_token_and_feature_edits_preserve_other_positions_and_checkpoint(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    dictionary = tmp_path / "dictionary.safetensors"
    tensors = {
        "encoder": torch.zeros(8, 1),
        "decoder": torch.ones(1, 8),
        "encoder_bias": torch.ones(1),
    }
    save_file(tensors, dictionary)
    baseline = LayerDebugger(model, inputs)
    baseline.step()
    debugger = LayerDebugger(model, inputs)
    debugger.command("patch-token zero 0 0 0")
    debugger.command(f'patch-feature feature 0 "{dictionary}" 0 2 -1')
    debugger.snapshot("edited")
    debugger.command("undo")
    assert set(debugger.edits) == {"zero"}
    debugger.command("redo")
    debugger.save(tmp_path / "edits.safetensors")
    restored = LayerDebugger(model, inputs)
    restored.load(tmp_path / "edits.safetensors")
    restored.step()
    assert torch.count_nonzero(restored.hidden[:, 0]) == 0
    assert torch.equal(restored.hidden[:, 1], baseline.hidden[:, 1])
    assert torch.equal(restored.hidden[:, -1], baseline.hidden[:, -1] + 1)
    debugger.restore("edited")
    debugger.step()
    assert torch.equal(debugger.hidden, restored.hidden)
    before = restored.hidden.clone()
    tensors["decoder"] *= 2
    save_file(tensors, dictionary)
    with pytest.raises(ValueError, match="checksum"):
        restored.load(tmp_path / "edits.safetensors")
    assert restored.index == 1 and torch.equal(restored.hidden, before)


def test_invalid_edit_and_failed_branch_do_not_change_state_or_leak_hooks() -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    debugger = LayerDebugger(model, {"input_ids": torch.tensor([[1, 2]])})
    for command in ["patch-head bad 0 2 0", "patch-token bad 0 2 0", "patch-token bad 0 0 nan"]:
        with pytest.raises(ValueError):
            debugger.command(command)
        assert debugger.edits == {} and debugger.undo_stack == []

    def fail(*args):
        raise RuntimeError("inference failure")

    handle = model.transformer.h[1].register_forward_hook(fail)
    before = debugger.hidden.clone()
    with pytest.raises(RuntimeError, match="inference failure"):
        debugger.compare("patch-head bad 0 1 0")
    handle.remove()
    assert debugger.index == 0 and debugger.edits == {} and debugger.undo_stack == []
    assert torch.equal(debugger.hidden, before)
    assert not model.transformer.h[0].attn.c_proj._forward_pre_hooks


def test_old_checkpoint_without_edits_remains_loadable(tmp_path: Path) -> None:
    from ldb.checkpoints import LayerCheckpoint, fingerprint, save_checkpoint

    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16)).eval()
    debugger = LayerDebugger(model, {"input_ids": torch.tensor([[1, 2]])})
    checkpoint = LayerCheckpoint(
        schema_version="ldb.layer-checkpoint/v1alpha1",
        identity=debugger.identity,
        next_layer=0,
        patches={},
        activation_hash=fingerprint(debugger.hidden),
    )
    save_checkpoint(tmp_path / "old.safetensors", checkpoint, debugger.hidden)
    debugger.command("patch-head remove 0 0 0")
    debugger.load(tmp_path / "old.safetensors")
    assert debugger.edits == {}


def test_next_token_regression_reports_groups_without_changing_paused_state(tmp_path: Path) -> None:
    import json

    from transformers import BatchEncoding

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

        def decode(self, ids):
            return str(ids)

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16)).eval()
    debugger = LayerDebugger(model, {"input_ids": torch.tensor([[1, 2]])}, tokenizer=Tokenizer())
    debugger.step()
    before = debugger.hidden.clone()
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": "ldb.suite/v1alpha1",
                "target": [{"case_id": "target", "prompt": "x", "target_token_id": 3}],
                "regression": [{"case_id": "control", "prompt": "y", "target_token_id": 4}],
            }
        )
    )
    report = debugger.command(f'regress "{suite}" "patch-head noop 0 1 1 -1"')
    assert report["groups"]["target"]["weighted_mean_delta"] == 0
    assert report["groups"]["regression"]["worst_delta"] == 0
    assert debugger.index == 1 and torch.equal(debugger.hidden, before)
    assert debugger.edits == {} and debugger.undo_stack == []
