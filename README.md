# LDB — Latent Debugger

> A runtime debugger and neural patching framework for large language models.

LDB brings `break`, `inspect`, `patch`, and `continue` to neural networks. It traces
internal representations during real inference, localizes causal failure mechanisms,
applies minimal request-scoped interventions, and verifies that a fix does not regress
unrelated behavior.

LDB is a CLI-first, local-first systems tool. It is not an LLM application observability
platform, and it does not claim to explain a model from correlations alone: observation,
attribution, and intervention evidence are kept distinct.

## Capabilities

**Inspect**

- Normalized decoder hooks across model families, offline through TransformerLens 3.x
  `TransformerBridge` or a native Hugging Face Transformers adapter.
- Q/K/V, attention scores and patterns, per-head Z/result activations, and pre-projection
  attention heads.
- Scalar hook summaries for every run, with selective full-tensor capture as `safetensors`
  in portable RunBundles.
- SAE dictionary loading with global and per-token feature activations.

**Intervene**

- Reversible scalar, tensor, head, SAE-feature, and expert-routing interventions.
- Request-scoped patches and request-filtered traces on live serving workers.
- Versioned Neural Patch documents that can be validated, exported, and replayed.

**Localize**

- Token-scoped causal ablation ranked by behavioral effect.
- Clean/corrupt activation patching for next-token localization.
- Directed causal propagation graphs from downstream activation recovery.
- One-command diagnosis: automatic localization followed by a bounded patch search.

**Verify**

- Regression-aware patch search over test suites, including heterogeneous
  multi-intervention search with an explicit no-op baseline.
- Reproducible export and verification of the selected patch.

**Attach to serving runtimes**

- Opt-in vLLM and SGLang plugins that stay dormant unless explicitly enabled.
- Bounded live traces, batch breakpoints, and an interactive attach shell over local
  Unix sockets.
- Linux hot attachment to already-running vLLM and SGLang CPython workers.
- Experimental fixed-address CUDA Graph control and summary-trace slots.

## Installation

LDB targets Python 3.11 and 3.12.

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
  --target-token-id 16046 --token -1 \
  --output causal-report.json

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

`ldb ablate` and `ldb attention` follow the same pattern. Offline analysis uses
TransformerLens by default; pass `--engine transformers` for the smaller native adapter.
`ldb run` records scalar summaries for every normalized hook point and writes full
tensors only for hooks matched by `--capture`.

## Attaching to a serving runtime

Install LDB into the same Python environment as the serving runtime. Plugins do nothing
unless `LDB_ENABLE=1` is set. The source-audited adapters accept vLLM `0.28.0` and
SGLang `0.5.19`; other versions fail closed unless `LDB_ALLOW_UNVERIFIED_RUNTIME=1` is
set deliberately for compatibility work. `ldb runtimes` prints the accepted versions.

Pin the model revision and tokenizer without loading weights:

```console
ldb identity <model> --output model-identity.json
```

Then start the server with the returned values:

```console
# vLLM
LDB_ENABLE=1 VLLM_PLUGINS=ldb LDB_INSTANCE_ID=production-a \
  LDB_MODEL_REVISION=<immutable-model-revision> \
  LDB_TOKENIZER_HASH=<ldb-tokenizer-hash> \
  vllm serve <model> --enforce-eager

# SGLang
LDB_ENABLE=1 SGLANG_PLUGINS=ldb LDB_INSTANCE_ID=production-a \
  LDB_MODEL_REVISION=<immutable-model-revision> \
  LDB_TOKENIZER_HASH=<ldb-tokenizer-hash> \
  sglang serve --model-path <model> --disable-cuda-graph
```

Attach from another terminal:

```console
ldb attach /tmp/ldb-$(id -u)/agents --instance production-a

(ldb) arm 'block.10.*'
(ldb) break stop-10 'block.10.residual.post' --timeout 60
(ldb) inspect-hit <hit-id> --capture
(ldb) patch-set suppress-block-10 'block.10.residual.post' scale 0.5 --request <request-id>
(ldb) continue
(ldb) patch-remove suppress-block-10
(ldb) disarm
```

To instrument a worker that is already running (Linux only):

```console
ldb hot-attach <worker-pid> --backend vllm \
  --model-revision <immutable-model-revision> \
  --tokenizer-hash <ldb-tokenizer-hash>
```

Hot attachment needs the same UID (or root), `gdb`, a C compiler, libc headers, an
audited backend version, and LDB's Python dependencies in the target interpreter. Repeat
it for every worker of a distributed runtime; ptrace policy may require
`CAP_SYS_PTRACE`.

### Execution modes

Request-scoped traces, full tensor capture, breakpoints, and tensor patches require an
eager execution path. With `LDB_CUDA_GRAPH=1`, scalar summaries and global scalar patches
also work during CUDA Graph replay; this mode is experimental.

### Fused-MoE routing

For models whose MoE layers never call a separate gate module, LDB hooks the router
itself and exposes `block.N.moe.router.topk_ids` and `topk_weights`. Where the router
uses a selection-only correction bias (DeepSeek `noaux_tc`), `block.N.moe.router.logits`
accepts `routing_bias` patches that change which experts are picked without changing
their weights. DeepSeek V4.1 decoder layers, whose residual stream is a multi-stream
hyper-connection, currently expose routing hooks only. This path requires eager execution.

### Security

The agent socket, descriptor, and captured tensors are user-owned with `0600`
permissions; socket and output directories use `0700`. Full tensor capture is
synchronous and meant for isolated debugging requests, not normal production traffic.

## Design principles

1. **Runtime** — inspect real model execution, not reconstructed diagrams.
2. **Causal** — distinguish observation and attribution from intervention evidence.
3. **Patch** — prefer minimal, request-scoped, reversible interventions.
4. **Regression** — a fix is incomplete until unrelated behavior is checked.
5. **Reproducibility** — every artifact is bound to model, tokenizer, backend, and
   hook-schema versions.

## Documentation

| Topic | Document |
|---|---|
| Architecture and roadmap | [docs/architecture-v0.1.md](docs/architecture-v0.1.md) |
| Runtime compatibility | [docs/compatibility.md](docs/compatibility.md) |
| GPU acceptance harness | [docs/gpu-validation.md](docs/gpu-validation.md) |
| TransformerLens integration | [docs/transformer-lens.md](docs/transformer-lens.md) |
| Patch search | [docs/patch-search.md](docs/patch-search.md) |
| Automatic diagnosis | [docs/automatic-diagnosis.md](docs/automatic-diagnosis.md) |
| Causal ablation | [docs/causal-ablation.md](docs/causal-ablation.md) |
| Causal propagation graphs | [docs/causal-graph.md](docs/causal-graph.md) |
| SAE dictionaries | [docs/sae-dictionaries.md](docs/sae-dictionaries.md) |
| Offline layer debugger | [docs/layer-debugger.md](docs/layer-debugger.md) |
| Model editing and training | [docs/model-editing-and-training.md](docs/model-editing-and-training.md) |
| Weights, gradients, and operators | [docs/weights-gradients-operators.md](docs/weights-gradients-operators.md) |
| Technology choices | [docs/technology.md](docs/technology.md) |
| MoE experiment tools (Chinese) | [docs/experiments/README.zh-CN.md](docs/experiments/README.zh-CN.md) |

`ldb experiment list` shows the bundled MoE experiment tools: expert fingerprints and
reversible expert interventions, packed mixed-precision workflows, MoE IR, and budget
planning. Their backends are optional dependencies.

## Status

LDB is in its initial implementation phase. Scheduler-isolated single-request
breakpoints are not implemented yet, and CUDA Graph support is experimental.

## Development

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[transformers,dev]'
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
.venv/bin/python scripts/generate_schemas.py
.venv/bin/python -m pip wheel --no-deps . -w dist
```

## License

LDB is released under the [MIT License](LICENSE).
