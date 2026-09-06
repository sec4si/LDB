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

- loading a Hugging Face causal language model;
- discovering normalized hook points for GPT-2 and Llama-family architectures;
- recording hook summaries during real inference;
- selectively capturing tensors as `safetensors`;
- inspecting portable RunBundles from the terminal;
- validating versioned Neural Patch documents.
- attaching the CLI to dormant in-process agents over local Unix sockets;
- loading opt-in SGLang and vLLM runtime plugins;
- recording bounded live traces and applying reversible scalar, tensor, and
  expert-routing interventions on eager execution paths;
- request-filtered traces, request-scoped patches, and bounded batch breakpoints;
- clean/corrupt activation patching for causal next-token localization;
- regression-aware patch search, reproducible export, and verification;
- opt-in fixed-address CUDA Graph control slots for global scalar patches.

Scheduler-isolated single-request breakpoints, multi-operation combinatorial search,
and CUDA Graph trace capture are not yet implemented. Live trace events and
request-scoped interventions currently require an eager execution path.

## Installation

LDB currently targets Python 3.11.

```console
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,transformers]'
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
  --output causal-report.json

ldb search sshleifer/tiny-gpt2 \
  --suite examples/suites/next-token.yaml \
  --hook 'block.*.residual.post' \
  --scale 0.5 --scale 0.75 --scale 1.25 \
  --output patch-search.json
```

`ldb run` records scalar summaries for every normalized hook point. Full tensors are
written only for hook names matched by `--capture`.

## Attach to a serving runtime

Install LDB in the same Python environment as the serving runtime. Runtime plugins
remain dormant unless `LDB_ENABLE=1` is set explicitly.

SGLang:

```console
LDB_ENABLE=1 SGLANG_PLUGINS=ldb \
  LDB_CUDA_GRAPH=1 \
  LDB_MODEL_REVISION=<immutable-model-revision> \
  LDB_TOKENIZER_HASH=<ldb-tokenizer-hash> \
  sglang serve --model-path <model> --disable-cuda-graph
```

vLLM:

```console
LDB_ENABLE=1 VLLM_PLUGINS=ldb \
  LDB_CUDA_GRAPH=1 \
  LDB_MODEL_REVISION=<immutable-model-revision> \
  LDB_TOKENIZER_HASH=<ldb-tokenizer-hash> \
  vllm serve <model> --enforce-eager
```

Attach from another terminal:

```console
ldb attach /tmp/ldb-$(id -u)/agents

(ldb) arm 'block.10.*'
(ldb) events
(ldb) break stop-10 'block.10.residual.post' --timeout 60
(ldb) status
(ldb) patch-set suppress-block-10 'block.10.residual.post' scale 0.5 \
  --request <request-id>
(ldb) continue
(ldb) patch-remove suppress-block-10
(ldb) disarm
```

The agent socket and descriptor files are user-owned and use `0600` permissions.
The parent directory uses `0700`. Full tensor capture is synchronous and intended
for isolated debugging requests, not normal production traffic.

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
