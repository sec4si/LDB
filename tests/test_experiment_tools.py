"""Portable experiment entry points and fingerprint controls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from typer.testing import CliRunner

from ldb.cli import app
from ldb.experiments.fingerprint import GroupedFingerprint
from ldb.experiments.response import compare


def test_registry_and_unknown_tool():
    runner = CliRunner()
    assert "precision-plan" in runner.invoke(app, ["experiment", "list"]).stdout
    assert runner.invoke(app, ["experiment", "run", "missing"]).exit_code != 0


def test_listing_does_not_import_backends():
    code = """
import sys
from typer.testing import CliRunner
from ldb.cli import app
assert CliRunner().invoke(app, ['experiment', 'list']).exit_code == 0
assert not {'torch', 'triton', 'mlx', 'numpy'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_response_cli_outside_repository(tmp_path):
    # Two identical experts: replacing one has exactly zero local error.
    outputs = np.array([[[1.0, 2.0], [3.0, 4.0]]] * 2)
    routes = np.array([[1.0, 0.0], [0.0, 1.0]])
    source = tmp_path / "responses.npz"
    target = tmp_path / "result.json"
    np.savez(source, outputs=outputs, routes=routes)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    command = [
        sys.executable,
        "-m",
        "ldb",
        "experiment",
        "run",
        "fingerprint",
        "--",
        "--input",
        str(source),
        "--output",
        str(target),
        "--remove",
        "1",
    ]
    subprocess.run(command, cwd=tmp_path, env=env, check=True)
    result = json.loads(target.read_text())
    assert len(set(result["mapping"])) == 1
    assert result["routing_weighted_error"] == [[0.0, 0.0], [0.0, 0.0]]
    assert subprocess.run(command, cwd=tmp_path, env=env, capture_output=True).returncode != 0


def test_response_routing_is_target_conditioned():
    outputs = np.array([[[0.0], [0.0]], [[2.0], [4.0]]])
    routes = np.array([[1.0, 0.0], [0.0, 0.5]])
    result = compare(outputs, routes)
    assert result["routing_weighted_error"][0][1] == 2.0
    assert result["routing_weighted_error"][1][0] == 2.0


def test_grouped_descriptor_preserves_permutation_and_function():
    generator = torch.Generator().manual_seed(12)
    gate, up = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64)
    down = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    x = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    original = GroupedFingerprint.from_weights(torch.cat([gate, up]), down)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = GroupedFingerprint.from_weights(
        torch.cat([gate[permutation], up[permutation]]), down[:, permutation]
    )
    assert original.discrepancy(changed) == 0.0
    expected = (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T
    torch.testing.assert_close(original(x), expected)


def test_precision_allocation_matches_exhaustive_search():
    import itertools

    from ldb.experiments.precision.qwen_mixed_loss_plan import choose

    groups = {
        "gu": [{"gain": x} for x in [3.0, 8.0, -2.0]],
        "down": [{"gain": x} for x in [2.0, 4.0]],
    }
    costs = {"gu": 3, "down": 2}
    items = [(row["gain"], costs[part]) for part, rows in groups.items() for row in rows]
    for budget in range(13):
        selected, gain = choose(groups, budget, costs)
        expected = max(
            sum(g * flag for (g, _), flag in zip(items, flags, strict=True))
            for flags in itertools.product([0, 1], repeat=len(items))
            if sum(c * flag for (_, c), flag in zip(items, flags, strict=True)) <= budget
        )
        assert gain == expected
        assert all(row["gain"] > 0 for row in selected)
