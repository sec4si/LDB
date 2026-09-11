# Budget-constrained model builder v1

Build actual weights under a target budget, evaluate a finite multi-host search,
confirm the selected candidate once and export an independently runnable model.
The planner never equates low byte count with retained capability.

The initial adapters are:

- native Torch TinyMoE checkpoint copies and cross-host expert replacement, with
  the existing MoE IR checking expert structure and checkpoint identity;
- original MLX Qwen1.5-MoE 4-bit weights and physically compacted expert banks,
  using frozen same-source calibration tables and nonuniform layer allocation.

The Qwen path is an explicit backend adapter; quantized Qwen has not been silently
promoted into the currently narrower Mixtral IR runtime. Existing source precision
is preserved; searching new precision levels, arbitrary architectures and
unrestricted expert synthesis are outside v1.

## Inputs and decisions

`spec.json` sets weight-file bytes, process physical-memory budget, acceptance
headroom, context length, generation length, single-request concurrency, task
accuracy/retention, text-loss retention, minimum decode throughput and finite
candidate count. `catalog.json` identifies hosts, interpreters, immutable weight/
config hashes, calibration tables and any predeclared foreign donor slots.

A candidate must pass all gates. A small host that cannot do the target tasks is
rejected. The ranking is task accuracy first, then bits/byte on the same text
prefixes, then weight bytes. This is the best observed candidate within a bounded
search, not a global optimum or a guarantee of general intelligence retention.

The compiler rounds Qwen removal to the measured four-expert increments and
allocates them across layers by DP on the frozen additive surrogate. Real combined
model measurements decide acceptance. It does not pad files to fake arbitrary
sizes. Output must be at or below the requested byte cap.

Development chooses a candidate. Confirmation cannot choose another candidate if
the winner fails. `result.json` explicitly reports `no_feasible_candidate`,
`confirmation_failed`, `standalone_failed`, or `delivered`. Failed candidates and
worker logs remain available. A delivery directory alone is not a success receipt.

## Run the local demonstration

Use the already model-enabled Torch and MLX environments. The catalog names each
backend interpreter. The root orchestrator needs the ordinary research environment;
the Qwen worker needs the `mlx` extra. No daemon or network model server is started.

```sh
python -m pytest tests/test_budget_builder.py
python -m ldb.experiments.budget_builder.prepare \
  --root runs/budget-builder-v1 \
  --legacy /path/to/original-research-checkout \
  --abi runs/expert-abi-v1 \
  --ir runs/moe-ir-v1
python -m ldb.experiments.budget_builder.run --root runs/budget-builder-v1
```

Preparation accepts `--spec /path/to/spec.json` and `--catalog /path/to/catalog.json`
to replace the demonstration settings. Use a new destination for every experiment.
The demo requires the pre-existing source-bound Qwen layer calibration and audited
TinyMoE inputs. It fetches new FineWeb-Edu prefixes and generates disjoint fixed
arithmetic/extraction prompts. An existing compact result is not substituted for
a newly constructed candidate.

The implemented resource monitor is **macOS only**. It samples process physical
footprint through `proc_pid_rusage` from cold start through exit, checks actual
weight-file bytes and executes full-length cache generation at the configured
context. It kills workers observed above the budget and applies 10% default
headroom. Sampling can miss transient peaks; this is not a kernel-enforced memory
cap or an arbitrary-concurrency service guarantee. MLX active allocation is not
used as a substitute for whole-process physical footprint. Building and baseline
measurement may require a larger machine than the target runtime budget.

Each worker runs in a fresh process, with a timeout, captured output and an explicit
exit/measurement receipt. Source hashes and the execution sources are checked
before work. The primary protocol is [here](../../docs/budget-builder-protocol.md).
The reused Qwen exporter/DP implementation is recorded in
`reused-source-provenance.json`; the original dirty checkout remains untouched.

## Delivery

On success `delivery/` contains actual standalone weights, tokenizer, backend
implementation when needed, `budget-run.py`, `budget_prompts.py` and
`budget-model.json` with the selected transform, verified limits and file hashes.
The delivered runner verifies these hashes and enforces the declared context and
generation limits.

```sh
/path/to/backend/python /path/to/delivery/budget-run.py \
  --task arithmetic --prompt 'What is 24 + 17? Output only the integer.' --max-tokens 16
```

Use `--task arithmetic` or `--task extract` to apply the exact tested task interface;
`--task general` uses ordinary native prompting and has no claim of equivalent task
accuracy. Source-host baselines use the same task interface. Evaluation results
cover these simple tasks and short text prefixes, not arbitrary application skills.

The independent-load check runs from `/tmp` with `PYTHONPATH` removed. Models and
raw downloaded text stay in ignored local runs. Public compact evidence contains
plans, numeric outcomes and manifests, not a copy of the model payload.

## Supply your own supported catalog and workload

The public entry point does not require the demonstration checkout paths. Give it
backend-compatible sources and two disjoint evaluation files:

```sh
PYTHONPATH=. python scripts/build_budget_model.py \
  --catalog catalog.json --spec spec.json \
  --development development.json --confirmation confirmation.json \
  --output runs/my-budget-build
```

Add `--plan-only` to freeze/inspect candidates without running them. Each evaluation
file contains `texts` and `tasks`; each task has `id`, `kind` (`arithmetic` or
`extract`), `prompt` and string `answer`. The existing demo JSON files are complete
format examples. Relative catalog asset paths resolve against the catalog file's
directory. Unsupported backends/task contracts are rejected, not silently treated
as tested configurations.

For the two-budget implementation trial, both v2 plans are frozen before execution
and reuse the exact same development/confirmation inputs from the first attempt.
The first attempt did not open confirmation because no candidate was selected.
Its missing external base-tokenizer path is retained as an implementation issue;
v2 pins and verifies native tokenizer files separately. Neither budget is changed
in response to its confirmation results. The 7.6 GiB failure remains a separate
reported result rather than being replaced by the 7.8 GiB demonstration.

## Runtime cache correction

The original 7.8 GiB confirmation passed quality but failed the 10% memory-headroom
gate. The v3 MLX worker and delivered runner cap **free allocator cache** at 256 MiB
before loading; this does not cap live tensors or change model weights. Its effect
is measured with the same failed workload, while a separate new confirmation set
is used for the final rebuilt pipeline. Weight and memory budgets and all quality
gates remain unchanged. `--fresh-confirmation-offset` with `--data-from` preserves
the development inputs while preparing disjoint fresh confirmation documents and
task prompts. The original failed receipts remain available.

## Input and delivery hardening

Accuracy thresholds now accept `1.0`; booleans are rejected as numeric settings.
The grader checks task IDs/kinds, duplicate IDs, correctness rows versus aggregate
accuracy, positive document token counts, finite metrics and an actual positive
memory observation before applying quality gates. Malformed measurements are an
incomplete evaluation rather than proof that no model is feasible.

Delivery is staged outside the source and published by directory rename only after
copied weight/tokenizer hashes and the byte budget pass. Python caches are excluded
from the immutable file contract. Existing destinations are refused and failed
staging directories are removed.

A previously confirmed model can be resealed without repeating model selection:

```sh
PYTHONPATH=. python scripts/repackage_budget_delivery.py \
  --run runs/budget-builder-v3-7p8 --output runs/budget-builder-packaging-v4
```

This validates historical frozen inputs and current grading, preserves the old
artifact, creates a new package, and checks independent loading with the memory
headroom gate. It does not claim a new capability experiment. Historical source
snapshots remain the reference for earlier measured runs; current source changes
must not be copied into those frozen snapshots.
