"""Evaluate a frozen finite search, confirm once, then deliver an actual model."""

import argparse
import os
from pathlib import Path

import numpy as np

from .common import freeze, read, sha, verify, write
from .delivery import export_delivery
from .resources import run_worker


def grade(process, baseline, spec):
    if process.get("returncode") != 0 or "result" not in process:
        return {"pass": False, "reasons": [process.get("reason") or "worker_failed"]}
    result = process["result"]
    try:
        for measured in [result, baseline]:
            tasks = measured["tasks"]
            ids = [t["id"] for t in tasks]
            if (
                not tasks
                or len(set(ids)) != len(ids)
                or any(type(t["correct"]) is not bool for t in tasks)
            ):
                raise ValueError("Invalid task rows")
            accuracy = sum(t["correct"] for t in tasks) / len(tasks)
            if (
                type(measured["task_accuracy"]) not in (int, float)
                or not np.isfinite(measured["task_accuracy"])
                or abs(measured["task_accuracy"] - accuracy) > 1e-12
            ):
                raise ValueError("Task aggregate disagrees with rows")
            for row in measured["documents"]:
                if (
                    type(row["tokens"]) is not int
                    or row["tokens"] <= 0
                    or type(row["nll"]) not in (int, float)
                    or not np.isfinite(row["nll"])
                    or row["nll"] < 0
                ):
                    raise ValueError("Invalid document measurement")
            for key in ["bits_per_byte", "weight_file_bytes", "decode_tps"]:
                if (
                    type(measured[key]) not in (int, float)
                    or not np.isfinite(measured[key])
                    or measured[key] < 0
                ):
                    raise ValueError("Invalid metric")
        if [(t["id"], t["kind"]) for t in result["tasks"]] != [
            (t["id"], t["kind"]) for t in baseline["tasks"]
        ]:
            raise ValueError("Task identities differ")
        if (
            not np.isfinite(process["peak_physical_footprint_bytes"])
            or process["peak_physical_footprint_bytes"] <= 0
        ):
            raise ValueError("Missing physical memory measurement")
    except (KeyError, TypeError, ValueError):
        return {"pass": False, "reasons": ["invalid_measurements"]}
    if (
        not result["documents"]
        or len(result["documents"]) != len(baseline["documents"])
        or [d["tokens"] for d in result["documents"]]
        != [d["tokens"] for d in baseline["documents"]]
    ):
        return {"pass": False, "reasons": ["incomparable_document_metrics"]}
    delta = np.array(
        [
            x["nll"] - y["nll"]
            for x, y in zip(result["documents"], baseline["documents"], strict=True)
        ]
    )
    if not np.isfinite(delta).all():
        return {"pass": False, "reasons": ["nonfinite_metrics"]}
    rng = np.random.default_rng(20260911)
    ci = np.quantile(
        delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1), [0.025, 0.975]
    ).tolist()
    gates = {
        "weight": result["weight_file_bytes"] <= spec["weight_bytes"],
        "memory": process["peak_physical_footprint_bytes"]
        <= spec["memory_bytes"] * (1 - spec["memory_headroom"]),
        "task_floor": result["task_accuracy"] >= spec["min_task_accuracy"],
        "task_retention": result["task_accuracy"]
        >= baseline["task_accuracy"] - spec["max_task_drop"],
        "text_retention": float(delta.mean()) <= spec["max_delta_nll"]
        and ci[1] <= spec["max_delta_nll_upper"],
        "speed": result["decode_tps"] >= spec["min_decode_tps"],
        "context": result["capacity_context_tokens"] == spec["context_tokens"],
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "delta_nll": float(delta.mean()),
        "interval95": ci,
        "reasons": [name for name, passed in gates.items() if not passed],
    }


def choose(rows):
    passed = [r for r in rows if r["grade"]["pass"]]
    if not passed:
        return None
    return min(
        passed,
        key=lambda r: (
            -r["process"]["result"]["task_accuracy"],
            r["process"]["result"]["bits_per_byte"],
            r["process"]["result"]["weight_file_bytes"],
        ),
    )["candidate"]["id"]


def execute(root):
    import platform

    if platform.system() != "Darwin":
        raise RuntimeError("Budget execution requires the macOS footprint monitor; use --plan-only")
    root = root.resolve()
    verify(root, "input-freeze.json")
    for name, expected in read(root / "input-freeze.json").items():
        if (
            name.startswith("sources/")
            and sha(Path(__file__).parents[3] / Path(name).relative_to("sources")) != expected
        ):
            raise ValueError("Execution sources changed since preparation")
    if (root / "selection.json").exists():
        raise FileExistsError("Frozen run already selected; use a new run directory")
    spec, catalog, plan = [read(root / name) for name in ["spec.json", "catalog.json", "plan.json"]]
    hosts = {h["id"]: h for h in catalog["hosts"]}
    by_id = {c["id"]: c for c in plan["candidates"]}
    for key, host in hosts.items():
        inv = plan["hosts"][key]
        for name, expected in inv["weight_sha256"].items():
            if sha(Path(host["path"]) / name) != expected:
                raise ValueError("Source checkpoint changed")
        for name, expected in host.get("tokenizer_sha256", {}).items():
            if sha(Path(host["tokenizer"]) / name) != expected:
                raise ValueError("Source tokenizer changed")
        if sha(Path(host["path"]) / "config.json") != inv["config_sha256"]:
            raise ValueError("Source config changed")
    environment = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).parents[3]),
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    records_dir = root / "measurements"
    records_dir.mkdir(exist_ok=True)

    def worker(stage, candidate, split="development", baseline=False):
        output = records_dir / f"{candidate['id']}-{stage}-{split}.json"
        if output.exists() or output.with_suffix(".process.json").exists():
            raise FileExistsError(output)
        command = [
            hosts[candidate["host"]]["python"],
            "-m",
            "ldb.experiments.budget_builder.worker",
            stage,
            "--root",
            str(root),
            "--candidate",
            candidate["id"],
            "--split",
            split,
            "--output",
            str(output),
        ]
        result = run_worker(
            command,
            Path.cwd(),
            output,
            None if baseline or stage == "build" else spec["memory_bytes"],
            spec["timeout_seconds"],
            environment,
        )
        print(stage, candidate["id"], split, result["returncode"], result.get("reason"), flush=True)
        return result

    baselines = {}
    for key in hosts:
        c = by_id[f"{key}-original"]
        baselines[key] = worker("evaluate", c, baseline=True)
    rows = []
    for candidate in plan["candidates"]:
        host = candidate["host"]
        if not candidate["preflight_pass"]:
            rows.append(
                {"candidate": candidate, "grade": {"pass": False, "reasons": ["budget_preflight"]}}
            )
            continue
        if "result" not in baselines[host]:
            rows.append(
                {
                    "candidate": candidate,
                    "grade": {"pass": False, "reasons": ["baseline_unavailable"]},
                }
            )
            continue
        if candidate["operation"] == "original":
            process = baselines[host]
        else:
            built = worker("build", candidate)
            if built["returncode"]:
                rows.append(
                    {"candidate": candidate, "grade": {"pass": False, "reasons": ["build_failed"]}}
                )
                continue
            process = worker("evaluate", candidate)
        outcome = grade(process, baselines[host]["result"], spec)
        rows.append({"candidate": candidate, "process": process, "grade": outcome})
        write(root / "development-partial.json", rows)
        print("gates", candidate["id"], outcome["pass"], outcome["reasons"], flush=True)
    write(root / "development-results.json", rows)
    winner = choose(rows)
    write(
        root / "selection.json",
        {
            "winner": winner,
            "ranking": [
                "task_accuracy descending",
                "bits_per_byte ascending",
                "weight bytes ascending",
            ],
            "scope": "Best measured candidate within this frozen finite search; no claim of global capability optimum",
        },
    )
    freeze(root, ["selection.json", "development-results.json"], "selection-freeze.json")
    if winner is None:
        incomplete = any(
            set(r["grade"]["reasons"])
            & {
                "baseline_unavailable",
                "build_failed",
                "worker_failed",
                "timeout",
                "invalid_measurements",
                "incomparable_document_metrics",
                "nonfinite_metrics",
            }
            for r in rows
        )
        write(
            root / "result.json",
            {
                "status": "incomplete_evaluation" if incomplete else "no_feasible_candidate",
                "winner": None,
                "delivered": False,
            },
        )
        return
    candidate = by_id[winner]
    baseline = worker(
        "evaluate", by_id[f"{candidate['host']}-original"], split="confirmation", baseline=True
    )
    confirmation = (
        worker("evaluate", candidate, split="confirmation")
        if candidate["operation"] != "original"
        else baseline
    )
    verdict = (
        grade(confirmation, baseline.get("result", {"documents": []}), spec)
        if "result" in baseline
        else {"pass": False, "reasons": ["confirmation_baseline_failed"]}
    )
    write(
        root / "confirmation-result.json",
        {"candidate": winner, "process": confirmation, "grade": verdict},
    )
    if not verdict["pass"]:
        write(
            root / "result.json",
            {
                "status": "confirmation_failed",
                "winner": winner,
                "delivered": False,
                "reason": "No reselection using confirmation outcomes",
            },
        )
        return
    source = (
        Path(hosts[candidate["host"]]["path"])
        if candidate["operation"] == "original"
        else root / "artifacts" / winner
    )
    destination = root / "delivery"
    delivery_host = dict(
        hosts[candidate["host"]], weight_sha256=plan["hosts"][candidate["host"]]["weight_sha256"]
    )
    export_delivery(source, destination, delivery_host, spec, candidate, verdict)
    # Verify copied model with a new process, outside the repository import path.
    output = records_dir / "standalone.json"
    command = [
        hosts[candidate["host"]]["python"],
        str(destination / "budget-run.py"),
        "--task",
        "arithmetic",
        "--prompt",
        "What is 24 + 17? Output only the integer.",
        "--max-tokens",
        "16",
        "--receipt",
        str(output),
    ]
    standalone_env = dict(os.environ)
    standalone_env.pop("PYTHONPATH", None)
    standalone = run_worker(
        command, Path("/tmp"), output, spec["memory_bytes"], spec["timeout_seconds"], standalone_env
    )
    delivered = (
        standalone["returncode"] == 0
        and standalone.get("result", {}).get("answer", "").strip() == "41"
    )
    write(
        root / "result.json",
        {
            "status": "delivered" if delivered else "standalone_failed",
            "winner": winner,
            "delivered": delivered,
            "delivery": str(destination),
            "standalone": standalone,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    execute(p.parse_args().root)
