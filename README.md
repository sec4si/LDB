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

Interactive breakpoints, causal replay, patch search, vLLM, and SGLang are planned
but are not yet implemented.

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
```

`ldb run` records scalar summaries for every normalized hook point. Full tensors are
written only for hook names matched by `--capture`.

## Design principles

1. **Runtime** — inspect real model execution, not reconstructed diagrams.
2. **Causal** — distinguish observation and attribution from intervention evidence.
3. **Patch** — prefer minimal, request-scoped, reversible interventions.
4. **Regression** — a fix is incomplete until unrelated behavior is checked.
5. **Reproducibility** — every artifact is bound to model, tokenizer, backend, and
   hook-schema versions.

See [the v0.1 architecture](docs/architecture-v0.1.md) for component boundaries and
the implementation roadmap.
