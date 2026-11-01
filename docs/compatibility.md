# Runtime Compatibility

LDB runtime adapters intentionally depend on a small set of serving-runtime
boundaries. The adapter must fail closed when these boundaries disappear instead of
silently tracing the wrong tensors.

The plugins refuse startup if their audited worker/model-forward boundaries are
missing. Continuing would otherwise advertise request-scoped traces and patches
without authoritative batch context. This explicit check is necessary for SGLang
because its hook registry logs target-resolution failures instead of propagating
them to the plugin loader.

Plugins also reject runtime package versions outside the audited stable rows below.
For deliberate adapter development against a new version, set
`LDB_ALLOW_UNVERIFIED_RUNTIME=1`; this bypasses only the version gate, not structural
boundary checks, and does not make that version supported or GPU verified.

## Source-audited snapshots

The following boundaries were checked against upstream source on 2026-09-07.

Offline analysis targets TransformerLens `3.8.x` through its recommended
`TransformerBridge` API. LDB uses raw Hugging Face weight semantics, disables the
Bridge's implicit BOS insertion for parity with serving prompts, and records the
TransformerLens version in every resulting artifact.

| Runtime snapshot | Upstream commit | Required boundaries |
|---|---|---|
| SGLang v0.5.19 | `0bcd822377da7b5718e674eaf9c870d349424dd1` | `sglang.srt.plugins`, `TpModelWorker.__init__`, `ModelRunner.forward`, `ForwardBatch.rids`, `ForwardBatch.extend_seq_lens_cpu` |
| SGLang | `2e8c03e2c71732b98ac29882ca1ee88bfa3fb1de` | `sglang.srt.plugins`, `TpModelWorker.__init__`, `ModelRunner.forward`, `ForwardBatch.rids`, `ForwardBatch.extend_seq_lens_cpu` |
| vLLM v0.28.0 | `2cf0a6915ce544dc493a0990f2ea38d81601128a` | `vllm.general_plugins`, V1 `Worker.load_model`, both V1/V2 `GPUModelRunner.execute_model` paths, `InputBatch.req_ids`, `SchedulerOutput.num_scheduled_tokens` |
| vLLM | `6865e67f0be02d53694517f6f71d7fb96492792d` | `vllm.general_plugins`, V1 `Worker.load_model`, `GPUModelRunner.execute_model`, `InputBatch.req_ids`, `SchedulerOutput.num_scheduled_tokens` |

The unversioned rows are upstream `main` snapshots. These are source audits, not
GPU integration-test claims. GPU acceptance must cover each supported runtime
release, model family, execution mode, and parallel topology before that
combination is listed as verified.

## Runtime matrix

2026-09-08: Qwen3-0.6B passed real single-GPU prepared/eager acceptance on
WSL2 + RTX 4070 for vLLM 0.28.0 (V1 runner, FlashAttention 2, PyTorch sampler)
and SGLang 0.5.19 (Triton attention, PyTorch sampler). Trace, global residual
ablation and deterministic restoration passed. This does not extend verification
to hot attachment, CUDA Graphs, distributed execution or other architectures.
See [the run record and configuration restrictions](wsl-gpu-acceptance.md).

2026-11-01: DeepSeek V4.1 Flash on 8× A800 with a vLLM 0.31.0 development build
(`vllm.v1.worker.gpu` runner, TP8 with expert parallelism, eager, prepared attachment,
version gate bypassed) passed fused-MoE routing traces, observe-only residual traces
with bit-identical armed and unarmed outputs, patch rejection on observe-only hooks, and
request context from the runner's `InputBatch`. Routing-bias patches, sequence
parallelism, CUDA Graphs and deep_gemm mega-mHC hardware were not covered. See
[the DeepSeek V4.1 record](deepseek-v41-gpu-validation.md).

| Capability | Transformers eager | SGLang eager | vLLM eager | CUDA Graph replay |
|---|---:|---:|---:|---:|
| Normalized module trace | Yes | Implemented | Implemented | Experimental summaries |
| Selective tensor capture | Yes | Implemented | Implemented | Not yet |
| Global scalar activation patch | N/A | Implemented | Implemented | Experimental |
| Global tensor activation patch | N/A | Implemented | Implemented | Not yet |
| Request-scoped activation patch | N/A | Implemented | Implemented | Not yet |
| Batch safe-point breakpoint | N/A | Implemented | Implemented | Not yet |
| Pre-projection head ablation | Yes | Implemented | Implemented | Not yet |
| Linux hot attachment | N/A | Implemented | Implemented | Not applicable |
| Visible unfused expert output | Implemented | Implemented | Implemented | Not yet |
| Single-request scheduler suspension | N/A | Not yet | Not yet | Not yet |

"Implemented" means the adapter and protocol path exist and have dependency-free
contract tests. It does not mean every model architecture or kernel backend has
passed a real GPU acceptance run.

## Decoder activation contracts

The native serving hooks bind the model's forward signature and select the floating
activation argument, including keyword calls. Positional tensors such as integer
RoPE positions must never be treated as residual activations.

The following eager decoder contracts were checked against the pinned runtime source:

| Runtime | Decoder class | Residual representation | Local validation |
|---|---|---|---|
| vLLM 0.28.0 | `LlamaDecoderLayer` | Deferred `(hidden_states, residual)` sum | CPU contract tests |
| vLLM 0.28.0 | `Qwen2DecoderLayer` | Deferred `(hidden_states, residual)` sum | CPU contract tests |
| vLLM 0.28.0 | `Qwen3DecoderLayer` | Deferred `(hidden_states, residual)` sum | CPU contract tests |
| SGLang 0.5.19 | `LlamaDecoderLayer` | Deferred `(hidden_states, residual)` sum | CPU contract tests |
| SGLang 0.5.19 | `Qwen2DecoderLayer` | Deferred `(hidden_states, residual)` sum | CPU contract tests |
| SGLang 0.5.19 | `Qwen3DecoderLayer` | Communicator-managed sum with additional residual input | CPU contracts for unsharded TP1/DP1/CP1 eager |
| vLLM 0.31.0 dev | DeepSeek V4.1 `DeepseekV4DecoderLayer` | Multi-stream hyper-connection; observe-only mean over streams from the next layer's aux capture | GPU validated, 8× A800 TP8 eager |

For these supported contracts, `block.N.residual.pre` means the logical input
sum, and `block.N.residual.post` means the logical output sum. The first output tensor
alone is generally the MLP contribution, not the complete residual stream. Trace-only
and no-op operations preserve the original pair. Interventions replace changed
elements by a `(modified, zero)` decomposition, preserving unmodified elements'
original split tensors so unrelated request rows are not unnecessarily rewritten.
Dormant hooks do not materialize the sum. SGLang Qwen3 also validates its communicator:
all layer modes must be `FULL` or `TP_ATTN_FULL`, communication group sizes must
be one, and sequence-parallel variants, reduce-scatter and fused all-reduce
quantization must be disabled. With an existing residual, `post_residual_addition`
contributes to the input sum. Interventions adjust it only at modified elements,
avoiding double addition. With no residual, the upstream behavior ignores that input.

This path requires matching floating tensor shapes, dtypes, and devices. Quantized
activation payloads, unknown decoder signatures with a `residual` argument, and
CUDA Graph mode for these deferred-residual decoders fail explicitly. The change
does not extend runtime version whitelists or validate distributed GPU execution, quantized
models, arbitrary Llama/Qwen derivatives, or actual GPU kernels. Named decoder
contracts are not blanket model-family certification. GPU acceptance remains required.

The normalized hook schema is now `ldb.hooks/v1alpha3`. Older reports remain
historical artifacts, but v1alpha2 patches must be regenerated and reverified before
loading into the new live agents: deferred-residual hook meaning has changed, so
silently reusing an old patch or merely editing its version field is not sufficient.

Pipeline-parallel `PPMissingLayer` modules are skipped during hook installation,
preserving global indices for real local layers. Group `arm` filters patterns per
rank; patch and breakpoint creation goes only to owning ranks. Other ranks return
an explicit `hook_not_local` skip. IDs stay unique across the group, and partial
failures roll back successful owners. Three local Unix-socket agents test routing
and rollback on CPU; this does not constitute multi-GPU pipeline acceptance.
The placeholder identities follow the defining modules, including SGLang's
`sglang.srt.layers.utils.common.PPMissingLayer` (the shorter `layers.utils` path
only re-exports it). Sources: [SGLang placeholder](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/layers/utils/common.py),
[vLLM placeholder](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/utils.py).

Source references: [vLLM Llama](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/llama.py),
[vLLM Qwen2](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/qwen2.py),
[vLLM Qwen3](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/qwen3.py),
[SGLang Llama](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/llama.py),
[SGLang Qwen2](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/qwen2.py),
[SGLang Qwen3](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/qwen3.py).
Its additional residual behavior follows the pinned
[LayerCommunicator](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/layers/communicator.py).

The shared-object injection mechanism has passed an isolated Linux acceptance test
for already-running vLLM-shaped and SGLang-shaped CPython workers, including agent
discovery and a live control-socket handshake. Real GPU server acceptance remains a
separate required gate.

"Experimental" means fixed-address graph control/trace slots and CPU contract tests
exist, but the path has not passed real CUDA Graph replay acceptance. Enable it
explicitly with `LDB_CUDA_GRAPH=1`. Patches support global scalar `scale`, `add`, and
`ablate` only. Traces expose the latest scalar summary and replay occurrence count;
they do not expose request IDs or full tensors.

Expert-output hooks are available only when experts remain visible as individual
PyTorch modules. Their rows contain only tokens routed to that expert and therefore
do not preserve the continuous batch's request-span layout. Global traces and
interventions are supported, but request-scoped traces, patches, token selectors,
and breakpoints fail closed on expert-output hooks. Router-logit hooks remain
token-aligned and request-addressable. For fused-MoE layers, LDB hooks the router and
exposes the selected expert IDs and weights; it does not claim visibility into the
expert computation inside an opaque fused-MoE kernel.

Pre-projection head ablation uses exact contiguous head slices. In tensor-parallel
serving, adapters publish authoritative `tp_rank` and `tp_size` values. LDB maps the
global head index to the owning rank, applies a local index there, and leaves other
ranks unchanged. Non-divisible head layouts or missing ownership metadata fail
closed.
