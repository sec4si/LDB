# GPU Acceptance

Source-contract tests cannot prove that hooks survive a serving runtime's model
wrapping, compilation, distributed topology, or CUDA Graph replay. LDB therefore
ships a real-server smoke harness rather than labeling source-audited combinations
as GPU verified.

```console
python scripts/gpu_smoke.py vllm facebook/opt-125m --mode eager
python scripts/gpu_smoke.py sglang facebook/opt-125m --mode eager
python scripts/gpu_smoke.py vllm facebook/opt-125m --mode graph
python scripts/gpu_smoke.py sglang facebook/opt-125m --mode graph
```

The eager test starts the OpenAI-compatible server, attaches through the private
agent directory, arms a normalized hook, sends a real completion, and requires a
trace event. The graph test requires `trace.cuda_graph` and `patch.cuda_graph`, arms
a normalized hook, and requires a replay-produced graph trace event. It then records
a deterministic baseline completion, installs a global ablation, and requires the
replayed completion or token log-probability to change. Merely seeing agent state
does not count as graph-execution evidence.

The `GPU smoke` GitHub workflow is manual and targets a self-hosted runner labeled
`self-hosted`, `linux`, `x64`, and `gpu`. A runtime combination is not GPU verified
until this workflow passes on the relevant runtime version, model family, mode, and
parallel topology.
