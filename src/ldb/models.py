"""Versioned domain models shared by the CLI, runtime, and artifact store."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HOOK_SCHEMA_VERSION: Final = "ldb.hooks/v1alpha1"
RUN_SCHEMA_VERSION: Final = "ldb.run/v1alpha1"
PATCH_SCHEMA_VERSION: Final = "ldb.patch/v1alpha1"


class FrozenModel(BaseModel):
    """Immutable base model for protocol and artifact values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceLevel(StrEnum):
    OBSERVED = "observed"
    CORRELATED = "correlated"
    ATTRIBUTED = "attributed"
    CAUSALLY_TESTED = "causally_tested"
    REPLICATED = "replicated"


class HookPhase(StrEnum):
    PRE = "pre"
    POST = "post"


class HookPoint(FrozenModel):
    name: str
    module_path: str
    phase: HookPhase


class TensorSummary(FrozenModel):
    shape: tuple[int, ...]
    dtype: str
    device: str
    mean: float | None = None
    std: float | None = None
    norm: float | None = None


class TraceEvent(FrozenModel):
    sequence: int = Field(ge=0)
    timestamp: datetime
    hook: str
    phase: HookPhase
    summary: TensorSummary
    tensor_file: str | None = None


class RunManifest(FrozenModel):
    schema_version: Literal["ldb.run/v1alpha1"] = "ldb.run/v1alpha1"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    tokenizer_id: str
    backend: str
    backend_version: str
    adapter: str
    adapter_version: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    prompt: str
    prompt_token_ids: tuple[int, ...]
    generation: dict[str, Any]
    output_text: str | None = None
    parent_run_id: str | None = None


class InterventionKind(StrEnum):
    SCALE = "scale"
    ABLATE = "ablate"
    REPLACE = "replace"
    ADD = "add"
    ROUTING_BIAS = "routing_bias"
    HEAD_ABLATE = "head_ablate"
    FEATURE_SCALE = "feature_scale"


class Intervention(FrozenModel):
    hook: str
    operation: InterventionKind
    value: float | None = None
    tensor: str | None = None
    expert: int | None = Field(default=None, ge=0)
    head: int | None = Field(default=None, ge=0)
    num_heads: int | None = Field(default=None, gt=0)
    dictionary: str | None = None
    feature: int | None = Field(default=None, ge=0)
    request_scoped: bool = True

    @model_validator(mode="after")
    def valid_arguments(self) -> Intervention:
        no_feature = self.dictionary is None and self.feature is None
        if self.operation is InterventionKind.SCALE:
            valid = (
                self.value is not None
                and self.tensor is None
                and self.expert is None
                and self.head is None
                and self.num_heads is None
                and no_feature
            )
        elif self.operation is InterventionKind.ADD:
            valid = (
                (self.value is None) != (self.tensor is None)
                and self.expert is None
                and self.head is None
                and self.num_heads is None
                and no_feature
            )
        elif self.operation is InterventionKind.REPLACE:
            valid = (
                self.tensor is not None
                and self.value is None
                and self.expert is None
                and self.head is None
                and self.num_heads is None
                and no_feature
            )
        elif self.operation is InterventionKind.ROUTING_BIAS:
            valid = (
                self.value is not None
                and self.tensor is None
                and self.expert is not None
                and self.head is None
                and self.num_heads is None
                and no_feature
            )
        elif self.operation is InterventionKind.HEAD_ABLATE:
            valid = (
                self.value is None
                and self.tensor is None
                and self.expert is None
                and self.head is not None
                and self.num_heads is not None
                and self.head < self.num_heads
                and no_feature
            )
        elif self.operation is InterventionKind.FEATURE_SCALE:
            valid = (
                self.value is not None
                and self.tensor is None
                and self.expert is None
                and self.head is None
                and self.num_heads is None
                and self.dictionary is not None
                and self.feature is not None
            )
        else:
            valid = (
                self.value is None
                and self.tensor is None
                and self.expert is None
                and self.head is None
                and self.num_heads is None
                and self.dictionary is None
                and self.feature is None
            )
        if not valid:
            raise ValueError(f"Invalid arguments for {self.operation.value} intervention")
        return self


class ValidationSummary(FrozenModel):
    target_success: float = Field(ge=0.0, le=1.0)
    related_generalization: float | None = Field(default=None, ge=0.0, le=1.0)
    unrelated_regression: float | None = None


class NeuralPatch(FrozenModel):
    schema_version: Literal["ldb.patch/v1alpha1"] = "ldb.patch/v1alpha1"
    patch_id: str
    description: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    adapter: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    interventions: tuple[Intervention, ...] = Field(min_length=1)
    validation: ValidationSummary | None = None
    evidence: EvidenceLevel = EvidenceLevel.OBSERVED
