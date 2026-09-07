# LDB — Latent Debugger

> A runtime debugger and neural patching framework for large language models.

LDB brings `break`, `inspect`, `patch`, and `continue` to neural networks. It traces
internal representations, localizes causal failure mechanisms, applies minimal
request-scoped interventions, and verifies behavioral regressions.

LDB is a CLI-first, local-first systems tool. It is not an LLM application
observability platform and it does not claim to explain a model from correlations
alone.

## Status

LDB is in its initial implementation phase. The first vertical slice supports:

- loading a Hugging Face causal language model through TransformerLens 3.x
  `TransformerBridge`, with a native Transformers fallback;
- normalizing stable decoder hooks exposed by TransformerLens across model families;
- inspecting Q/K/V, attention scores and patterns, per-head Z/result activations,
  and applying semantically axis-safe token/head interventions;
- recording hook summaries during real inference;
- selectively capturing tensors as `safetensors`;
- inspecting portable RunBundles from the terminal;
- validating versioned Neural Patch documents;
- attaching the CLI to dormant in-process agents over local Unix sockets;
- loading opt-in SGLang and vLLM runtime plugins;
- recording bounded live traces and applying reversible scalar, tensor, and
  expert-routing interventions on eager execution paths;
- pre-projection attention-head inspection and exact head ablation;
- request-filtered traces, request-scoped patches, and bounded batch breakpoints;
- token-scoped causal ablation ranked by behavioral effect;
- clean/corrupt activation patching for causal next-token localization;
- directed causal propagation graphs from downstream activation recovery;
- canonical SAE dictionary loading and top-feature inspection;
- reporting global and per-token SAE feature activations;
- request-scoped single-feature SAE scaling;
- regression-aware patch search, reproducible export, and verification;
- bounded heterogeneous multi-intervention search with an explicit no-op;
- automatic failure localization followed by regression-aware minimal patch search;
- Linux hot attachment to already-running vLLM and SGLang CPython workers;
- opt-in fixed-address CUDA Graph control and summary-trace slots.

Scheduler-isolated single-request breakpoints are not yet implemented. CUDA Graph
summary traces and global scalar patches are experimental; request-scoped traces,
full tensor capture, and breakpoints require an eager execution path.

## Installation

LDB currently targets Python 3.11 and 3.12.

```console
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,interpretability]'
```

## Quick start

```console
ldb run sshleifer/tiny-gpt2 \
  --prompt "The Eiffel Tower is in" \
  --max-new-tokens 1 \
  --capture 'block.*.residual.post'

ldb inspect runs/<run-id>
ldb patch validate examples/patches/scale-feature.yaml

ldb causal sshleifer/tiny-gpt2 \
  --clean-prompt "The Eiffel Tower is in" \
  --corrupt-prompt "The Eiffel Tower is on" \
  --target-token-id 16046 \
  --token -1 \
  --output causal-report.json

ldb ablate sshleifer/tiny-gpt2 \
  --prompt "The Eiffel Tower is in" \
  --target-token-id 16046 \
  --token -1 \
  --output ablation-report.json

ldb attention sshleifer/tiny-gpt2 \
  --prompt "The Eiffel Tower is in" \
  --output attention-report.json

ldb diagnose sshleifer/tiny-gpt2 \
  --clean-prompt "The Eiffel Tower is in" \
  --failing-prompt "The Eiffel Tower is on" \
  --target-token-id 16046 \
  --output diagnosis-report.json \
  --patch-output candidate-patch.yaml

ldb search sshleifer/tiny-gpt2 \
  --suite examples/suites/next-token.yaml \
  --hook 'block.*.residual.post' \
  --scale 0.5 --scale 0.75 --scale 1.25 \
  --output patch-search.json
```

Offline analysis uses TransformerLens by default. Pass `--engine transformers` to
use LDB's smaller native Hugging Face adapter instead.

`ldb run` records scalar summaries for every normalized hook point. Full tensors are
written only for hook names matched by `--capture`.

## Attach to a serving runtime

Install LDB in the same Python environment as the serving runtime. Runtime plugins
remain dormant unless `LDB_ENABLE=1` is set explicitly.
The current source-audited adapters accept vLLM `0.28.0` and SGLang `0.5.19`.
Other versions fail closed unless an operator deliberately sets
`LDB_ALLOW_UNVERIFIED_RUNTIME=1` for compatibility development.
Run `ldb runtimes` to print the accepted version set from the installed LDB build.

Resolve the immutable model revision and canonical tokenizer hash without loading
model weights:

```console
ldb identity <model> --output model-identity.json
```

Use the returned `model_revision` and `tokenizer_hash` values for
`LDB_MODEL_REVISION` and `LDB_TOKENIZER_HASH` below.

SGLang:

```console
LDB_ENABLE=1 SGLANG_PLUGINS=ldb LDB_INSTANCE_ID=production-a \
  LDB_MODEL_REVISION=<immutable-model-revision> \
  LDB_TOKENIZER_HASH=<ldb-tokenizer-hash> \
  sglang serve --model-path <model> --disable-cuda-graph
```

vLLM:

```console
LDB_ENABLE=1 VLLM_PLUGINS=ldb LDB_INSTANCE_ID=production-a \
  LDB_MODEL_REVISION=<immutable-model-revision> \
  LDB_TOKENIZER_HASH=<ldb-tokenizer-hash> \
  vllm serve <model> --enforce-eager
```

For experimental summaries and global scalar patches during CUDA Graph replay, set
`LDB_CUDA_GRAPH=1` and omit `--disable-cuda-graph` / `--enforce-eager`. Full tensor
capture, breakpoints, tensor patches, and request-scoped controls still require
eager mode.

Attach from another terminal:

```console
ldb attach /tmp/ldb-$(id -u)/agents --instance production-a

(ldb) arm 'block.10.*'
(ldb) hooks 'block.10.*'
(ldb) events
(ldb) break stop-10 'block.10.residual.post' --timeout 60
(ldb) status
(ldb) inspect-hit <hit-id> --capture
(ldb) patch-set suppress-block-10 'block.10.residual.post' scale 0.5 \
  --request <request-id>
(ldb) continue
(ldb) patch-remove suppress-block-10
(ldb) disarm
```

On Linux, an already-running worker can be instrumented without a server restart:

```console
ldb hot-attach <worker-pid> --backend vllm \
  --model-revision <immutable-model-revision> \
  --tokenizer-hash <ldb-tokenizer-hash>
ldb hot-attach <worker-pid> --backend sglang \
  --model-revision <immutable-model-revision> \
  --tokenizer-hash <ldb-tokenizer-hash>
```

Hot attachment requires the same UID (or root), `gdb`, a C compiler, libc headers,
an audited backend version, and LDB's Python dependencies in the target interpreter.
Repeat it for every worker process in a distributed runtime. Linux ptrace policy may
require `CAP_SYS_PTRACE` or an equivalent operator-approved setting.

The agent socket, descriptor, and captured tensor files are user-owned and use
`0600` permissions. Socket and live-output directories use `0700`. Full tensor
capture is synchronous and intended for isolated debugging requests, not normal
production traffic.

## Design principles

1. **Runtime** — inspect real model execution, not reconstructed diagrams.
2. **Causal** — distinguish observation and attribution from intervention evidence.
3. **Patch** — prefer minimal, request-scoped, reversible interventions.
4. **Regression** — a fix is incomplete until unrelated behavior is checked.
5. **Reproducibility** — every artifact is bound to model, tokenizer, backend, and
   hook-schema versions.

See [the v0.1 architecture](docs/architecture-v0.1.md) for component boundaries and
the implementation roadmap. See [runtime compatibility](docs/compatibility.md) for
the distinction between implemented adapter paths and GPU-verified combinations.
See [patch search](docs/patch-search.md) for the optimization objective and suite
format.
See [SAE dictionaries](docs/sae-dictionaries.md) for the canonical tensor layout and
feature-inspection command.
See [technology choices](docs/technology.md) for language, protocol, storage, and
dependency decisions.
See [GPU acceptance](docs/gpu-validation.md) for the real-server smoke harness and
the evidence required before a runtime combination is marked verified.
See [causal propagation graphs](docs/causal-graph.md) for the directed-edge
experiment and its evidence boundary.
See [causal ablation](docs/causal-ablation.md) for token-level removal experiments
and effect interpretation.
See [TransformerLens integration](docs/transformer-lens.md) for the offline engine
boundary and normalized hook mapping.
See [automatic diagnosis](docs/automatic-diagnosis.md) for the one-command causal
localization and bounded patch-search workflow.
