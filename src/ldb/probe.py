"""Linear probes on residual-stream activations, with a random-label control.

A probe that reads a property only shows that the property is linearly decodable from a
layer, not that the model uses it. The control trains an identical probe on shuffled
training labels; accuracy above that control is what the representation contributes.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from ldb.models import FrozenModel, HookPoint
from ldb.runtime import ProbeCallback, ProbeHandle

_RESIDUAL_HOOK = re.compile(r"^block\.(?P<layer>\d+)\.residual\.post$")


class ProbeRuntime(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    revision: str | None

    def hook_points(self) -> Sequence[HookPoint]: ...

    def install_probe(self, hook: HookPoint, callback: ProbeCallback) -> ProbeHandle: ...

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float: ...


class ProbeExample(FrozenModel):
    text: str
    label: str


class ProbeLayer(FrozenModel):
    layer: int = Field(ge=0)
    accuracy: float
    control_accuracy: float
    selectivity: float


class ProbeReport(FrozenModel):
    schema_version: Literal["ldb.probe/v1alpha1"] = "ldb.probe/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    labels: tuple[str, ...]
    position: Literal["last", "mean"]
    train_examples: int = Field(ge=1)
    test_examples: int = Field(ge=1)
    majority_baseline: float
    controls: int = Field(ge=1)
    seed: int
    layers: tuple[ProbeLayer, ...]


def load_examples(path: Path) -> list[ProbeExample]:
    """Read JSON Lines with one {"text": ..., "label": ...} object per line."""

    lines = path.read_text(encoding="utf-8").splitlines()
    return [ProbeExample.model_validate(json.loads(line)) for line in lines if line.strip()]


def train_probes(
    runtime: ProbeRuntime,
    examples: Sequence[ProbeExample],
    *,
    layers: Sequence[int] | None = None,
    position: Literal["last", "mean"] = "last",
    test_fraction: float = 0.3,
    controls: int = 5,
    seed: int = 0,
    l2: float = 1e-3,
) -> ProbeReport:
    """Fit one linear probe per residual.post layer and compare it with shuffled labels."""

    import torch

    labels = tuple(sorted({example.label for example in examples}))
    if len(labels) < 2:
        raise ValueError("Probing needs at least two distinct labels")
    if not 0 < test_fraction < 1 or controls < 1:
        raise ValueError("test_fraction must be in (0, 1) and controls must be positive")
    features = _collect(runtime, examples, layers, position)
    targets = torch.tensor([labels.index(example.label) for example in examples])

    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    split = round(len(order) * (1 - test_fraction))
    train, test = order[:split], order[split:]
    if not train or not test:
        raise ValueError("Too few examples for the requested train/test split")

    generator = torch.Generator().manual_seed(seed)
    train_targets, test_targets = targets[train], targets[test]
    results = []
    for layer in sorted(features):
        values = features[layer]

        def score(fit_targets: Any, values: Any = values) -> float:
            return _fit_and_score(values[train], fit_targets, values[test], test_targets, l2)

        accuracy = score(train_targets)
        shuffles = [torch.randperm(len(train), generator=generator) for _ in range(controls)]
        control = sum(score(train_targets[shuffle]) for shuffle in shuffles) / controls
        results.append(
            ProbeLayer(
                layer=layer,
                accuracy=accuracy,
                control_accuracy=control,
                selectivity=accuracy - control,
            )
        )

    majority = Counter(int(targets[index]) for index in train).most_common(1)[0][0]
    return ProbeReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        labels=labels,
        position=position,
        train_examples=len(train),
        test_examples=len(test),
        majority_baseline=sum(int(targets[index]) == majority for index in test) / len(test),
        controls=controls,
        seed=seed,
        layers=tuple(results),
    )


def _collect(
    runtime: ProbeRuntime,
    examples: Sequence[ProbeExample],
    layers: Sequence[int] | None,
    position: str,
) -> dict[int, Any]:
    import torch

    hooks = {
        int(match.group("layer")): hook
        for hook in runtime.hook_points()
        if (match := _RESIDUAL_HOOK.match(hook.name))
        and (layers is None or int(match.group("layer")) in layers)
    }
    if not hooks:
        raise ValueError("Runtime exposes no residual.post hooks for the selected layers")
    rows: dict[int, list[Any]] = {layer: [] for layer in hooks}

    def capture(hook: HookPoint, tensor: Any) -> None:
        match = _RESIDUAL_HOOK.match(hook.name)
        assert match is not None
        sequence = tensor[0].detach().float().cpu()
        rows[int(match.group("layer"))].append(
            sequence[-1] if position == "last" else sequence.mean(0)
        )

    handles = [runtime.install_probe(hook, capture) for hook in hooks.values()]
    try:
        for example in examples:
            runtime.next_token_log_probability(example.text, 0)
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.stack(values) for layer, values in rows.items()}


def _fit_and_score(
    train_values: Any, train_targets: Any, test_values: Any, test_targets: Any, l2: float
) -> float:
    """Multinomial logistic regression on standardized features, scored on held-out data."""

    import torch

    mean = train_values.mean(0)
    scale = train_values.std(0).clamp_min(1e-6)
    inputs, held_out = (train_values - mean) / scale, (test_values - mean) / scale
    classes = int(max(train_targets.max(), test_targets.max())) + 1
    weights = torch.zeros(inputs.shape[1], classes, requires_grad=True)
    bias = torch.zeros(classes, requires_grad=True)
    optimizer: Any = torch.optim.LBFGS([weights, bias], max_iter=200, line_search_fn="strong_wolfe")

    def loss() -> Any:
        optimizer.zero_grad()
        value: Any = torch.nn.functional.cross_entropy(inputs @ weights + bias, train_targets)
        value = value + l2 * weights.square().sum()
        value.backward()
        return value

    optimizer.step(loss)
    with torch.no_grad():
        predictions = (held_out @ weights + bias).argmax(-1)
        return float((predictions == test_targets).float().mean())
