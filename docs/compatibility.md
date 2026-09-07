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

| Capability | Transformers eager | SGLang eager | vLLM eager | CUDA Graph replay |
|---|---:|---:|---:|---:|
| Normalized module trace | Yes | Implemented | Implemented | Experimental summaries |
| Selective tensor capture | Yes | Implemented | Implemented | Not yet |
| Global scalar activation patch | N/A | Implemented | Implemented | Experimental |
| Global tensor activation patch | N/A | Implemented | Implemented | Not yet |
| Request-scoped activation patch | N/A | Implemented | Implemented | Not yet |
| Batch safe-point breakpoint | N/A | Implemented | Implemented | Not yet |
| Pre-projection head ablation | Yes | Implemented | Implemented | Not yet |
| Visible unfused expert output | Implemented | Implemented | Implemented | Not yet |
| Single-request scheduler suspension | N/A | Not yet | Not yet | Not yet |

"Implemented" means the adapter and protocol path exist and have dependency-free
contract tests. It does not mean every model architecture or kernel backend has
passed a real GPU acceptance run.

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
token-aligned and request-addressable. LDB does not claim visibility inside an
opaque fused-MoE kernel.

Pre-projection head ablation uses exact contiguous head slices. In tensor-parallel
serving, adapters publish authoritative `tp_rank` and `tp_size` values. LDB maps the
global head index to the owning rank, applies a local index there, and leaves other
ranks unchanged. Non-divisible head layouts or missing ownership metadata fail
closed.
