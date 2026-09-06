from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ldb.models import (
    EvidenceLevel,
    HookPhase,
    HookPoint,
    Intervention,
    InterventionKind,
    NeuralPatch,
)
from ldb.search import BehaviorCase, EvaluationSuite
from ldb.verification import verify_patch


class Handle:
    def __init__(self, runtime: VerificationRuntime) -> None:
        self.runtime = runtime

    def remove(self) -> None:
        self.runtime.intervention = None


class VerificationRuntime:
    model_id = "fake-model"
    revision = "revision"
    backend = "fake"
    backend_version = "1"
    tokenizer_hash = "tokenizer-hash"

    def __init__(self) -> None:
        self.hook = HookPoint(
            name="block.0.residual.post", module_path="block", phase=HookPhase.POST
        )
        self.intervention: Any = None

    def hook_points(self) -> tuple[HookPoint, ...]:
        return (self.hook,)

    def install_probe(self, hook: HookPoint, callback: Any) -> Handle:
        raise AssertionError("not used")

    def install_intervention(self, hook: HookPoint, callback: Any) -> Handle:
        self.intervention = callback
        return Handle(self)

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        value = 1.0
        if self.intervention is not None:
            value = float(self.intervention(self.hook, value))
        return -((value - (0.5 if prompt == "target" else 1.0)) ** 2)


def test_verify_patch_checks_target_and_regression_thresholds(tmp_path: Path) -> None:
    patch = NeuralPatch(
        patch_id="patch-1",
        description="Scale one residual stream.",
        model_id="fake-model",
        model_revision="revision",
        tokenizer_hash="tokenizer-hash",
        adapter="fake",
        interventions=(
            Intervention(
                hook="block.0.residual.post",
                operation=InterventionKind.SCALE,
                value=0.75,
                request_scoped=False,
            ),
        ),
        evidence=EvidenceLevel.CAUSALLY_TESTED,
    )
    suite = EvaluationSuite(
        target=(
            BehaviorCase(
                case_id="failure", prompt="target", target_token_id=1, category="locality"
            ),
        ),
        related=(
            BehaviorCase(
                case_id="related",
                prompt="target",
                target_token_id=1,
                category="generalization",
            ),
        ),
        regression=(
            BehaviorCase(
                case_id="control", prompt="control", target_token_id=2, category="benchmark"
            ),
        ),
    )

    report = verify_patch(
        VerificationRuntime(),
        patch,
        suite,
        patch_directory=tmp_path,
        target_min_gain=0.1,
        regression_max_drop=0.1,
    )

    assert report.passed is True
    assert report.target_success == 1.0
    assert report.related_generalization == 1.0
    assert report.unrelated_regression == -0.0625
    assert report.categories["locality"].success == 1.0
    assert report.categories["generalization"].case_count == 1
    assert report.categories["benchmark"].worst_delta == -0.0625


def test_verify_patch_rejects_tokenizer_mismatch(tmp_path: Path) -> None:
    patch = NeuralPatch(
        patch_id="patch-1",
        description="Mismatched tokenizer.",
        model_id="fake-model",
        model_revision="revision",
        tokenizer_hash="wrong-hash",
        adapter="fake",
        interventions=(
            Intervention(
                hook="block.0.residual.post",
                operation=InterventionKind.SCALE,
                value=1.0,
                request_scoped=False,
            ),
        ),
    )
    suite = EvaluationSuite(
        target=(BehaviorCase(case_id="failure", prompt="target", target_token_id=1),)
    )
    with pytest.raises(ValueError, match="tokenizer hash"):
        verify_patch(VerificationRuntime(), patch, suite, patch_directory=tmp_path)
