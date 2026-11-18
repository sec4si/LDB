from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from ldb.cli import app
from ldb.generation import GenerationInput, GenerationResult, GenerationSettings
from ldb.generation_evaluation import (
    GenerationCase,
    GenerationSuite,
    OutputRule,
    evaluate_generation,
    evaluate_rules,
    load_generation_suite,
)
from ldb.models import HookPhase, HookPoint, Intervention, NeuralPatch


def result(text="SAFE", reason="eos", raw=None):
    return GenerationResult(
        rendered_prompt="SECRET exists only in the input",
        prompt_token_ids=(1, 2),
        generated_token_ids=(3, 0),
        output_text=text,
        raw_output_text=raw if raw is not None else text + "<eos>",
        raw_content_text=raw.removesuffix("<eos>") if raw is not None else text,
        finish_reason=reason,
        generation_config={"do_sample": False},
    )


def suite():
    return GenerationSuite(
        cases=(
            GenerationCase(
                case_id="attack",
                group="attack",
                category="same-category",
                input=GenerationInput(prompt="attack"),
                rules=(OutputRule(kind="not_contains", value="SECRET"),),
            ),
            GenerationCase(
                case_id="benign",
                group="benign",
                category="same-category",
                input=GenerationInput(prompt="benign"),
                rules=(OutputRule(kind="exact", value="42"),),
            ),
        )
    )


class FakeRuntime:
    model_id = "fake"
    revision = "revision"
    backend = "transformers"
    backend_version = "test"
    adapter_version = "test"
    tokenizer_hash = "tokenizer"
    device = "cpu"

    def __init__(self):
        self.active = False
        self.calls = []
        self.fail_generation = False
        self.interrupt = False
        self.hook = HookPoint(
            name="block.0.residual.post", module_path="layer", phase=HookPhase.POST
        )

    def hook_points(self):
        return (self.hook,)

    def install_intervention(self, hook, callback):
        self.active = True
        runtime = self

        class Handle:
            def remove(self):
                runtime.active = False

        return Handle()

    def generate_result(self, request, *, settings):
        self.calls.append((request.prompt, self.active))
        if self.active and self.interrupt:
            raise KeyboardInterrupt
        if self.active and self.fail_generation:
            raise RuntimeError("generation failed")
        text = (
            ("SAFE" if self.active else "SECRET")
            if request.prompt == "attack"
            else ("refuse" if self.active else "42")
        )
        return result(text)


def patch():
    return NeuralPatch(
        patch_id="example",
        description="test",
        model_id="fake",
        model_revision="revision",
        tokenizer_hash="tokenizer",
        adapter="ldb.adapters.transformers",
        interventions=(Intervention(hook="block.0.residual.post", operation="scale", value=0.5),),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": "x", "messages": [{"role": "user", "content": "x"}]},
        {"prompt": ""},
        {"messages": [{"role": "assistant", "content": "x"}]},
    ],
)
def test_input_requires_unambiguous_nonempty_input(payload):
    with pytest.raises(ValidationError):
        GenerationInput.model_validate(payload)


def test_generation_settings_reject_sampling_and_invalid_limits():
    for data in ({"do_sample": True}, {"max_new_tokens": 0}, {"num_beams": 2}):
        with pytest.raises(ValidationError):
            GenerationSettings.model_validate(data)


def test_suite_requires_unique_ids_and_normal_task_control():
    original = suite()
    with pytest.raises(ValidationError, match="unique"):
        GenerationSuite(cases=(original.cases[0], original.cases[0]))
    with pytest.raises(ValidationError, match="both attack and benign"):
        GenerationSuite(
            cases=(original.cases[0], original.cases[0].model_copy(update={"case_id": "other"}))
        )
    with pytest.raises(ValidationError):
        OutputRule(kind="not_contains", value="")


def test_prompt_is_not_scored_as_generated_leak():
    verdict, _ = evaluate_rules(result(), (OutputRule(kind="not_contains", value="SECRET"),))
    assert verdict == "pass"


@pytest.mark.parametrize("reason", ["length", "unknown"])
@pytest.mark.parametrize(
    "kind,value", [("exact", "SAFE"), ("contains", "SAFE"), ("not_contains", "SECRET")]
)
def test_incomplete_generation_never_passes(reason, kind, value):
    verdict, _ = evaluate_rules(result(reason=reason), (OutputRule(kind=kind, value=value),))
    assert verdict == "inconclusive"


def test_witnessed_violation_fails_even_when_truncated_or_hidden_in_raw_text():
    for generated in (result("SECRET", "length"), result("SAFE", raw="SECRET<eos>")):
        verdict, _ = evaluate_rules(generated, (OutputRule(kind="not_contains", value="SECRET"),))
        assert verdict == "fail"


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_answer_is_not_a_safety_success(text):
    verdict, _ = evaluate_rules(result(text), (OutputRule(kind="not_contains", value="SECRET"),))
    assert verdict == "inconclusive"


def test_all_rules_are_required_and_case_is_preserved():
    verdict, rules = evaluate_rules(
        result("Safe"),
        (
            OutputRule(kind="not_contains", value="SECRET"),
            OutputRule(kind="contains", value="SAFE"),
        ),
    )
    assert verdict == "fail"
    assert [rule.verdict for rule in rules] == ["pass", "fail"]


def test_patch_improvement_does_not_hide_benign_regression(tmp_path):
    runtime = FakeRuntime()
    report = evaluate_generation(runtime, suite(), patch=patch(), patch_directory=tmp_path)
    assert runtime.active is False
    assert runtime.calls == [
        ("attack", False),
        ("benign", False),
        ("attack", True),
        ("benign", True),
    ]
    assert report.paired_changes["attack"].fixed == 1
    assert report.paired_changes["benign"].regressed == 1
    assert report.groups["attack"]["patched"].pass_rate == 1
    assert report.groups["benign"]["patched"].failure_rate == 1
    assert report.categories["attack"]["same-category"]["patched"].total == 1
    assert report.categories["benign"]["same-category"]["patched"].total == 1
    assert report.passed is False
    assert len(report.suite_sha256) == 64
    assert report.patch == patch()
    assert json.loads(report.model_dump_json())["cases"][0]["baseline"]["generation"]


def test_generation_errors_remain_in_denominators_and_cleanup(tmp_path):
    runtime = FakeRuntime()
    runtime.fail_generation = True
    report = evaluate_generation(runtime, suite(), patch=patch(), patch_directory=tmp_path)
    assert runtime.active is False
    assert report.passed is False
    assert report.groups["attack"]["patched"].inconclusive_rate == 1
    assert report.paired_changes["benign"].unresolved == 1
    assert report.cases[0].patched.error == "RuntimeError: generation failed"


def test_keyboard_interrupt_removes_patch_and_propagates(tmp_path):
    runtime = FakeRuntime()
    runtime.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        evaluate_generation(runtime, suite(), patch=patch(), patch_directory=tmp_path)
    assert runtime.active is False


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("model_id", "wrong", "model ID"),
        ("model_revision", "wrong", "revision"),
        ("tokenizer_hash", "wrong", "tokenizer"),
        ("adapter", "wrong", "adapter"),
        ("hook_schema_version", "wrong", "hook schema"),
    ],
)
def test_patch_identity_fails_before_generation(tmp_path, field, value, message):
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match=message):
        evaluate_generation(
            runtime,
            suite(),
            patch=patch().model_copy(update={field: value}),
            patch_directory=tmp_path,
        )
    assert runtime.calls == []
    assert not runtime.active


def test_unresolved_revision_and_token_scoped_patch_are_rejected(tmp_path):
    runtime = FakeRuntime()
    runtime.revision = None
    with pytest.raises(ValueError, match="revision"):
        evaluate_generation(runtime, suite(), patch=patch(), patch_directory=tmp_path)
    runtime.revision = "revision"
    operation = patch().interventions[0].model_copy(update={"token": -1})
    with pytest.raises(ValueError, match="Token-scoped"):
        evaluate_generation(
            runtime,
            suite(),
            patch=patch().model_copy(update={"interventions": (operation,)}),
            patch_directory=tmp_path,
        )
    assert runtime.calls == []


def test_missing_hook_patch_installation_fails(tmp_path):
    operation = Intervention(hook="block.999.residual.post", operation="scale", value=1.0)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="did not match"):
        evaluate_generation(
            runtime,
            suite(),
            patch=patch().model_copy(update={"interventions": (operation,)}),
            patch_directory=tmp_path,
        )
    assert not runtime.active


def test_suite_roundtrip_and_hash_change_with_rules(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(suite().model_dump_json())
    loaded = load_generation_suite(path)
    assert loaded == suite()
    first = evaluate_generation(FakeRuntime(), loaded)
    changed_case = loaded.cases[0].model_copy(
        update={"rules": (OutputRule(kind="not_contains", value="OTHER"),)}
    )
    second = evaluate_generation(
        FakeRuntime(), GenerationSuite(cases=(changed_case, loaded.cases[1]))
    )
    assert first.suite_sha256 != second.suite_sha256
    assert first.passed is False
    assert second.passed is True


def test_cli_writes_full_report_and_returns_failure(tmp_path, monkeypatch):
    source = tmp_path / "suite.json"
    source.write_text(suite().model_dump_json())
    output = tmp_path / "report.json"
    monkeypatch.setattr("ldb.cli._load_offline_adapter", lambda *a, **kw: FakeRuntime())
    cli = CliRunner().invoke(
        app, ["evaluate-generation", "fake", "--suite", str(source), "--output", str(output)]
    )
    assert cli.exit_code == 1, cli.output
    report = json.loads(output.read_text())
    assert report["groups"]["attack"]["baseline"]["failed"] == 1
    assert report["cases"][0]["baseline"]["generation"]["output_text"] == "SECRET"
    assert "Literal-rule" in cli.output
    second = CliRunner().invoke(
        app, ["evaluate-generation", "fake", "--suite", str(source), "--output", str(output)]
    )
    assert second.exit_code == 2
    assert json.loads(output.read_text()) == report


def test_cli_baseline_success_and_invalid_suite(tmp_path, monkeypatch):
    source = tmp_path / "suite.json"
    original = suite()
    changed = GenerationSuite(
        cases=(
            original.cases[0].model_copy(
                update={"rules": (OutputRule(kind="not_contains", value="OTHER"),)}
            ),
            original.cases[1],
        )
    )
    source.write_text(changed.model_dump_json())
    monkeypatch.setattr("ldb.cli._load_offline_adapter", lambda *a, **kw: FakeRuntime())
    cli = CliRunner().invoke(
        app,
        [
            "evaluate-generation",
            "fake",
            "--suite",
            str(source),
            "--output",
            str(tmp_path / "pass.json"),
        ],
    )
    assert cli.exit_code == 0, cli.output
    source.write_text('{"cases": []}')
    cli = CliRunner().invoke(
        app,
        [
            "evaluate-generation",
            "fake",
            "--suite",
            str(source),
            "--output",
            str(tmp_path / "invalid.json"),
        ],
    )
    assert cli.exit_code != 0
    assert not (tmp_path / "invalid.json").exists()


def test_example_suite_is_valid():
    path = Path(__file__).resolve().parents[1] / "examples/suites/generation-safety.yaml"
    assert {case.group for case in load_generation_suite(path).cases} == {"attack", "benign"}


def test_terminal_eos_text_is_not_payload_but_interior_control_text_is():
    rule = (OutputRule(kind="not_contains", value="eos"),)
    assert evaluate_rules(result("SAFE"), rule)[0] == "pass"
    # Only the actual terminal token is omitted, not a preceding payload token.
    generated = result("SAFE", raw="SAFE<eos><eos>")
    assert evaluate_rules(generated, rule)[0] == "fail"


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_token_ids": (1, 5)},
        {"rendered_prompt": "time-dependent input"},
        {"chat_template_sha256": "changed-template"},
        {"generation_config": {"do_sample": False, "max_new_tokens": 64}},
    ],
)
def test_input_or_config_drift_prevents_paired_attribution(tmp_path, change):
    class DriftingRuntime(FakeRuntime):
        def generate_result(self, request, *, settings):
            output = result("SAFE" if request.prompt == "attack" else "42")
            return output.model_copy(update=change) if self.active else output

    runtime = DriftingRuntime()
    report = evaluate_generation(runtime, suite(), patch=patch(), patch_directory=tmp_path)
    assert not report.passed
    assert not runtime.active
    for group in ("attack", "benign"):
        assert report.groups[group]["patched"].pass_rate == 1
        assert report.paired_changes[group].fixed == 0
        assert report.paired_changes[group].regressed == 0
        assert report.paired_changes[group].unresolved == 1
    assert all(case.comparison_error is not None for case in report.cases)
    assert all(case.baseline.generation and case.patched.generation for case in report.cases)


def test_partial_installation_interrupt_rolls_back_existing_hooks(tmp_path, monkeypatch):
    from ldb.offline import install_interventions

    runtime = FakeRuntime()
    loaded = []

    def load_resource(operation, directory):
        loaded.append(operation)
        if len(loaded) == 2:
            raise KeyboardInterrupt
        return None

    monkeypatch.setattr("ldb.offline._load_resource", load_resource)
    with pytest.raises(KeyboardInterrupt):
        install_interventions(runtime, patch().interventions * 2, resource_directory=tmp_path)
    assert len(loaded) == 2
    assert not runtime.active


def test_cleanup_attempts_every_handle_even_after_an_error():
    from ldb.offline import remove_interventions

    removed = []

    class Handle:
        def __init__(self, number):
            self.number = number

        def remove(self):
            removed.append(self.number)
            if self.number == 2:
                raise RuntimeError("cleanup failed")

    with pytest.raises(ExceptionGroup, match="cleanup failed") as caught:
        remove_interventions([Handle(1), Handle(2), Handle(3)])
    assert removed == [3, 2, 1]
    assert isinstance(caught.value.exceptions[0], RuntimeError)


def resource_patch():
    return patch().model_copy(
        update={
            "interventions": (
                Intervention(
                    hook="block.0.residual.post", operation="replace", tensor="tensor.bin"
                ),
            )
        }
    )


def test_missing_resource_fails_before_any_generation(tmp_path):
    runtime = FakeRuntime()
    with pytest.raises(FileNotFoundError):
        evaluate_generation(runtime, suite(), patch=resource_patch(), patch_directory=tmp_path)
    assert runtime.calls == []
    assert not runtime.active


def test_external_resource_content_is_identified_in_report(tmp_path, monkeypatch):
    import hashlib

    content = b"synthetic resource bytes; loader is a test double"
    (tmp_path / "tensor.bin").write_bytes(content)
    monkeypatch.setattr("ldb.offline._load_resource", lambda *args: None)
    report = evaluate_generation(
        FakeRuntime(), suite(), patch=resource_patch(), patch_directory=tmp_path
    )
    assert report.patch_resource_sha256 == {"tensor.bin": hashlib.sha256(content).hexdigest()}


@pytest.mark.parametrize("change_in_patch", [False, True])
def test_resource_drift_is_rejected_and_hooks_are_removed(tmp_path, monkeypatch, change_in_patch):
    resource = tmp_path / "tensor.bin"
    resource.write_bytes(b"original")
    monkeypatch.setattr("ldb.offline._load_resource", lambda *args: None)

    class ChangingRuntime(FakeRuntime):
        def generate_result(self, request, *, settings):
            output = super().generate_result(request, settings=settings)
            if self.active == change_in_patch:
                resource.write_bytes(b"changed")
            return output

    runtime = ChangingRuntime()
    with pytest.raises(ValueError, match="resources changed"):
        evaluate_generation(runtime, suite(), patch=resource_patch(), patch_directory=tmp_path)
    assert not runtime.active


def test_cli_rejects_invalid_parent_before_loading_model(tmp_path, monkeypatch):
    source = tmp_path / "suite.json"
    source.write_text(suite().model_dump_json())
    parent = tmp_path / "parent-is-a-file"
    parent.write_text("preserve")
    calls = []
    monkeypatch.setattr("ldb.cli._load_offline_adapter", lambda *a, **kw: calls.append(True))
    cli = CliRunner().invoke(
        app,
        [
            "evaluate-generation",
            "fake",
            "--suite",
            str(source),
            "--output",
            str(parent / "report.json"),
        ],
    )
    assert cli.exit_code == 2, cli.output
    assert calls == []
    assert parent.read_text() == "preserve"
