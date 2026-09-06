# Technology Choices

## Language

Python 3.11 and 3.12 are the implementation languages for the initial system. LDB
must execute inside PyTorch-based vLLM and SGLang workers, so a separate native core
would add an FFI boundary without removing the Python integration requirement.

Rust is not part of the current runtime. It remains an option for a future remote
transport or high-volume artifact index only after profiling proves Python is the
bottleneck. The model hot path stays in tensor operations executed by PyTorch and
the serving engine.

## Components

| Boundary | Choice | Reason |
|---|---|---|
| CLI | Typer, Rich, prompt-toolkit | Typed commands, terminal tables, and an interactive debugger loop without a frontend. |
| Runtime integration | PyTorch module hooks | The narrowest common boundary across Transformers, vLLM, and SGLang eager execution. |
| Serving plugins | `vllm.general_plugins`, `sglang.srt.plugins` | Prepared attachment through supported startup extension points. |
| Control plane | Versioned JSON Lines over Unix sockets | Local, inspectable, dependency-light, and protected by filesystem ownership and `0600` permissions. |
| Domain validation | Pydantic | Rejects unknown protocol and artifact fields and emits JSON Schema. |
| Tensor storage | safetensors | Non-executable, portable tensor artifacts with explicit names and shapes. |
| Human-authored artifacts | YAML | Reviewable Neural Patch, suite, and candidate documents. |
| Machine reports | JSON / JSON Lines | Deterministic parsing for traces, causal reports, search, and verification. |
| SAE boundary | Canonical safetensors dictionary | Avoids runtime coupling to one interpretability library's checkpoint classes. |
| Quality gates | Ruff, strict mypy, pytest | Formatting, static contracts, and focused behavioral tests. |
| CI | GitHub Actions, Python 3.11/3.12 | Verifies the supported interpreter range and wheel construction. |

## Deliberate exclusions

- No web UI or application-level prompt observability layer.
- No arbitrary process injection. The serving process must preload the dormant LDB
  plugin; later attach and detach do not restart model weights.
- No pickle for model-generated artifacts.
- No permanent weight mutation in the initial architecture.
- No claim that Python hooks can stop execution inside a fused CUDA kernel.

## Execution modes

Eager mode supports trace events, tensor capture, breakpoints, request filters, and
all intervention types. Experimental CUDA Graph control records fixed-address
scale/bias operations before graph capture and supports global scalar `scale`,
`add`, and `ablate`. Other graph features remain unavailable until they have a
static-buffer representation and real GPU acceptance evidence.
