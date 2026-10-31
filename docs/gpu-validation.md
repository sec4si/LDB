# GPU Acceptance

Completed runs: [WSL2 / RTX 4070 single-GPU eager acceptance, 2026-09-08](wsl-gpu-acceptance.md).
The record includes vLLM 0.28.0 and SGLang 0.5.19 with Qwen3-0.6B,
required environment settings, failure logs and successful intervention/restoration evidence.

Source-contract tests cannot prove that hooks survive a serving runtime's model
wrapping, compilation, distributed topology, or CUDA Graph replay. LDB therefore
ships a real-server smoke harness rather than labeling source-audited combinations
as GPU verified.

```console
python scripts/gpu_smoke.py vllm facebook/opt-125m --mode eager --artifacts evidence/vllm-eager
python scripts/gpu_smoke.py sglang facebook/opt-125m --mode eager --artifacts evidence/sglang-eager
python scripts/gpu_smoke.py vllm facebook/opt-125m --mode graph --artifacts evidence/vllm-graph
python scripts/gpu_smoke.py sglang facebook/opt-125m --mode graph --artifacts evidence/sglang-graph
```

To verify true side-loading rather than startup plugin registration, use hot
attachment. The harness starts the server with every LDB activation variable removed,
waits until its HTTP API is ready, discovers the serving CPython workers, injects LDB,
and only then sends trace and patch traffic:

```console
python scripts/gpu_smoke.py vllm facebook/opt-125m --mode eager --attachment hot --artifacts evidence/vllm-hot
python scripts/gpu_smoke.py sglang facebook/opt-125m --mode eager --attachment hot --artifacts evidence/sglang-hot
```

Hot attachment is eager-only: hooks cannot be inserted retroactively into CUDA Graphs
captured before injection. Both hot and prepared eager tests require a normalized
trace and a `patched_summary` carrying the patch ID from a subsequent real HTTP
request. The patch must change completion choices or token log probability; removal
must restore deterministic baseline choices. Missing effects or restoration fail
acceptance, and raw evidence is retained for diagnosis.
The self-hosted runner must permit same-UID `ptrace` or grant `CAP_SYS_PTRACE`, and
must provide `gdb`, a C compiler, and libc development headers.

The eager test starts the OpenAI-compatible server, attaches through the private
agent directory, arms a normalized hook, sends a real completion, and requires a
trace event. The graph test requires `trace.cuda_graph` and `patch.cuda_graph`, arms
a normalized hook, and requires a replay-produced graph trace event. It then records
a deterministic baseline completion, installs a global ablation, and requires the
replayed completion or token log-probability to change. Merely seeing agent state
does not count as graph-execution evidence.

Graph acceptance also checks only ranks owning the selected hook, requires replay
events before and after intervention, and verifies deterministic restoration after
removing the patch. Ordinary eager trace events cannot satisfy this gate.
`graph-evidence.json` retains owner inventories, responses, events, live patch
records and restoration results. Missing evidence from any owning rank fails.

Eager evidence is checked on ranks that actually own `block.0.residual.post`;
pipeline stages that do not execute that layer are not required to emit its event.
`eager-evidence.json` records the hook inventories, owning result indices, original
HTTP responses, trace/patch events and restoration result. Owner indices follow the
rank-sorted agent array in `result.json`. Every owning rank must produce evidence.

For the newly adapted SGLang Qwen3 contract, start with ordinary single-GPU eager
execution; do not enable sequence/context parallelism or quantized activations:

```console
python scripts/gpu_smoke.py sglang Qwen/Qwen3-0.6B --mode eager --attachment prepared --artifacts evidence/sglang-qwen3-eager
python scripts/gpu_smoke.py vllm Qwen/Qwen3-0.6B --mode eager --attachment prepared --artifacts evidence/vllm-qwen3-eager
```

These are acceptance commands, not records of completed GPU runs. The harness
starts an isolated server on its configured port and shuts it down on completion.

The `GPU smoke` GitHub workflow is manual and targets a self-hosted runner labeled
`self-hosted`, `linux`, `x64`, and `gpu`. A runtime combination is not GPU verified
until this workflow passes on the relevant runtime version, model family, mode, and
parallel topology.

```console
gh workflow run "GPU smoke" \
  --repo sec4si/LDB \
  -f backend=vllm \
  -f runtime_version=0.28.0 \
  -f model=facebook/opt-125m \
  -f mode=eager \
  -f attachment=hot
```

The workflow requires an exact runtime package version instead of installing an
unbounded latest release. Every run uploads the server log and, on success, a
versioned `result.json` containing the LDB commit, backend version, model, execution
mode, attachment path, injected PIDs, Python/PyTorch/CUDA versions, GPU identities,
trace and patch evidence, and digests of baseline and patched responses. A green job
without this evidence bundle does not qualify a combination as verified.
