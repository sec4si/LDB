"""Generate committed JSON Schemas from LDB's Pydantic contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from ldb.causal import CausalGraphReport, CausalReport
from ldb.live import (
    AgentDescriptor,
    AgentRequest,
    AgentResponse,
    ArmConfig,
    BreakpointHit,
    BreakpointInspection,
    LiveBreakpoint,
    LiveIntervention,
    LiveTraceEvent,
)
from ldb.models import NeuralPatch, RunManifest
from ldb.sae import FeatureReport
from ldb.search import (
    CandidateSet,
    CombinationSearchReport,
    EvaluationSuite,
    SearchReport,
)
from ldb.verification import VerificationReport

SCHEMAS: dict[str, type[BaseModel]] = {
    "agent-descriptor": AgentDescriptor,
    "agent-request": AgentRequest,
    "agent-response": AgentResponse,
    "arm-config": ArmConfig,
    "breakpoint-hit": BreakpointHit,
    "breakpoint-inspection": BreakpointInspection,
    "candidate-set": CandidateSet,
    "causal-graph-report": CausalGraphReport,
    "causal-report": CausalReport,
    "evaluation-suite": EvaluationSuite,
    "feature-report": FeatureReport,
    "live-breakpoint": LiveBreakpoint,
    "live-intervention": LiveIntervention,
    "live-trace-event": LiveTraceEvent,
    "neural-patch": NeuralPatch,
    "run-manifest": RunManifest,
    "search-report": SearchReport,
    "combination-search-report": CombinationSearchReport,
    "verification-report": VerificationReport,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(exist_ok=True)
    expected = {f"{name}.schema.json" for name in SCHEMAS}
    for path in root.glob("*.schema.json"):
        if path.name not in expected:
            path.unlink()
    for name, model in SCHEMAS.items():
        path = root / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
