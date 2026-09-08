# Runtime Attachment

## Attachment contract

Scripted `ldb attach <target> -e <command>` performs the protocol handshake but
prints only the command result as JSON. Interactive attachment also prints the
handshake. This keeps scripted output directly consumable by JSON parsers.

LDB supports two attachment paths. Prepared attachment loads a dormant runtime
plugin at startup. Linux hot attachment uses `gdb` to load a temporary shared object
into an already-running CPython worker; its constructor creates a new thread, waits
for the GIL, discovers the existing model runner, and installs the same runtime
agent without restarting the server.

```console
ldb hot-attach <worker-pid> --backend vllm \
  --model-revision <immutable-model-revision> \
  --tokenizer-hash <ldb-tokenizer-hash>
ldb hot-attach <worker-pid> --backend sglang \
  --model-revision <immutable-model-revision> \
  --tokenizer-hash <ldb-tokenizer-hash>
```

Hot attachment is intentionally explicit and local. It verifies process ownership,
requires an audited runtime version, and fails if it cannot identify a supported
live runner. It requires Linux, `gdb`, a C compiler plus libc headers, and ptrace
permission. Repeat the command for each worker PID. The injected bootstrap reports
the exact agent socket only after hook installation succeeds.

For prepared attachment, run `ldb identity <model>` before server startup. It resolves the requested Hugging
Face revision to an immutable commit and hashes the tokenizer graph, special-token
map, and chat template using the same function as the offline Transformers adapter.
Pass those values as `LDB_MODEL_REVISION` and `LDB_TOKENIZER_HASH`; Neural Patch
loading fails closed when they are absent or different. The agent also publishes
its normalized hook-schema version, which must match the patch.

```text
ldb CLI
   |
   | JSON lines over a user-private Unix socket
   v
LDB agent in each inference worker
   |
   | normalized dormant PyTorch probes
   v
model runner
```

Every tensor-parallel or pipeline-parallel worker publishes a descriptor in the
agent directory. `ldb attach <directory>` discovers live sockets and routes
hook-scoped commands to owning ranks; lifecycle and listing commands reach every
rank. When multiple serving runtimes share the same Unix user and
agent directory, give each server a stable `LDB_INSTANCE_ID` and select it with
`ldb attach <directory> --instance <id>`. LDB refuses an ambiguous directory instead
of risking a command against the wrong model server.

## Protocol

The current protocol version is `ldb.agent/v1alpha1`. Each request and response is
one bounded JSON object followed by a newline. Supported methods are:

| Method | Purpose |
|---|---|
| `hello` | Negotiate protocol, backend, topology, and capabilities. |
| `status` | Read arm state, event count, run ID, and active patches. |
| `hooks` | List installed normalized hooks, optionally filtered by a glob. |
| `arm` | Select hook globs and optionally enable full tensor persistence. |
| `disarm` | Stop trace recording without unloading probes. |
| `events` | Read bounded trace events after a sequence number. |
| `clear` | Clear buffered events. |
| `patch.set` | Install a reversible eager-path activation intervention. |
| `patch.remove` | Remove an intervention by patch ID. |
| `patch.list` | List active interventions. |
| `break.set` | Stop a model thread at a normalized eager hook. |
| `break.remove` | Remove a breakpoint by ID. |
| `break.list` | List configured breakpoints. |
| `break.inspect` | Inspect a stopped activation and optionally persist its tensor. |
| `continue` | Release one or all stopped model threads. |

`patch.set` supports scalar `scale`, `add`, and `ablate`; tensor-backed `add`
and `replace`; and an expert-specific `routing_bias`. Tensor files are loaded from
the serving worker's filesystem and must contain either one tensor or a tensor named
`activation`.

```console
(ldb) patch-set steer-1 'block.20.residual.post' add \
  --tensor /srv/ldb/steering.safetensors --request request-42
(ldb) patch-set route-1 'block.20.moe.router.logits' routing_bias -0.25 \
  --expert 7 --request request-42
(ldb) patch-set head-1 'block.20.attention.heads' head_ablate \
  --head 17 --heads 32 --request request-42
(ldb) patch-set feature-1 'block.20.residual.post' feature_scale 0.4 \
  --dictionary /srv/ldb/block-20-sae.safetensors --feature 18372 \
  --token -1 --request request-42
(ldb) patch-load neural-patch.yaml --request request-42
```

`patch-load` validates the versioned Neural Patch, checks its model ID against every
connected worker, requires exact model revision and tokenizer hash matches, resolves
tensor paths relative to the patch document, and rolls back interventions already
installed if a later intervention or rank fails. A request-scoped intervention
cannot be loaded without at least one `--request`. Set `LDB_MODEL_REVISION` and
`LDB_TOKENIZER_HASH` at server startup when the serving engine does not expose those
immutable identities from the loaded model.

`--token INDEX` narrows an intervention to one token within every selected request;
negative indices count from that request's end. Live token selectors require at
least one `--request` so continuous batches cannot be indexed ambiguously.

Request-scoped controls require token-aligned hook rows. Residual, attention, MLP,
and router-logit hooks retain the continuous batch layout. Individual expert-output
hooks contain only routed tokens, so request-scoped traces, patches, and breakpoints
targeting them are rejected instead of guessing an invalid row mapping.

Tensor-parallel attention projections receive local head shards. Live `head_ablate`
uses adapter-provided `tp_rank`/`tp_size` metadata to translate one global head index
on the owning rank; other ranks retain the global patch record but execute a no-op.
Missing ownership metadata and non-divisible global head counts fail closed.

Patch and breakpoint IDs are immutable while installed. Remove an existing ID
before reusing it. Reversible multi-rank commands snapshot prior state before
mutation and restore successful ranks if another rank fails.

For pipeline parallelism, non-local `PPMissingLayer` entries do not advertise hooks.
Group `arm` clears trace selection on ranks without a local match. Patch/breakpoint
creation on non-owning ranks returns `{"skipped": true, "reason": "hook_not_local"}`
in that rank's result slot. Every requested pattern must match at least one attached
rank. Direct commands reject non-local hooks when a rank has an installed hook
inventory. IDs are checked across the group before creation, including non-targeted
ranks; a failure rolls back only ranks changed by the command.

When attaching to an agent directory, LDB performs a live handshake, requires one
coherent backend/model identity, rejects duplicate ranks, and requires the complete
`0..world_size-1` rank set before issuing commands. The Unix-socket control plane is
therefore single-host in v0.1; a distributed deployment spanning hosts must expose
all workers through a deliberately colocated control plane before group attachment
is safe. All ranks must also advertise the same capability set and TP/PP/DP sizes,
whose product must equal `world_size` when topology metadata is available.
Instance IDs contain only letters, digits, dots, underscores, and hyphens, begin
with a letter or digit, and are limited to 128 characters. They are routing labels,
not authentication secrets.

Unknown fields are rejected. Socket requests are limited to 1 MiB and responses to
16 MiB. Bulk tensors are written as `safetensors` instead of crossing the control
socket.

For prepared model runtimes, `arm`, `patch.set`, and `break.set` reject every hook
glob that matches no installed hook. Use `hooks` before setting controls instead of
discovering misspellings after a silent no-op.

Decoder pre-hooks bind `hidden_states` by its forward argument name rather than
assuming the first positional tensor is an activation. The audited Llama/Qwen
deferred-residual contracts normalize the logical sum of hidden and residual
branches at decoder boundaries. Their eager-only support and explicit exclusions
are listed in [decoder activation contracts](compatibility.md#decoder-activation-contracts).

## Runtime integration

Hot attachment patches the already-loaded runner class's execution boundary before
installing model hooks. Consequently, subsequent batches receive the same request
context propagation as prepared attachment. Existing CUDA Graphs do not contain
newly injected hooks; use eager mode for complete hot-attached tracing and patching.

### SGLang

The `sglang.srt.plugins` entry point installs LDB after
`TpModelWorker.__init__`. It uses the loaded model's `named_modules()` topology and
registers normalized probes on decoder blocks, attention modules, MLP modules, and
visible MoE gates.

Python hooks do not execute during CUDA Graph replay. Operators must use
`--disable-cuda-graph` for complete live traces. With `LDB_CUDA_GRAPH=1`, LDB installs
fixed-address scalar control and summary slots before graph capture; captured
operations read control buffers and update summary buffers during replay even though
the Python hook does not run.

### vLLM

The `vllm.general_plugins` entry point wraps the V1 GPU worker's model-load boundary
and installs the same normalized probes. The plugin is loaded only when both
`LDB_ENABLE=1` and `VLLM_PLUGINS=ldb` are present.

`--enforce-eager` is currently required for request-aware or full-tensor traces. The
agent reports `execution.eager_only` for those capabilities. `LDB_CUDA_GRAPH=1`
additionally reports `trace.cuda_graph` and `patch.cuda_graph`, enabling fixed-address
summary and scalar control slots before vLLM captures the model.

Graph control composes global scalar `scale`, `add`, and `ablate` patches in install
order. It rejects request filters, tensor replacement, and expert routing because
those operations require dynamic layouts that the current graph slot does not
represent. Updating a slot synchronizes its device before the control call returns.

Graph summary slots compute mean, population standard deviation, and norm on device,
then increment a fixed device counter on every replay. `arm` snapshots those counters;
later `events` calls synchronize only the slot outputs and emit the most recent
summary plus the number of occurrences since the last poll. Graph trace events are
labeled `source: cuda_graph`. Request filtering and full tensor capture are rejected
instead of silently degrading.

## Current safety boundary

The SGLang adapter propagates `ForwardBatch.rids` and prefill/decode token spans
through the model forward. The vLLM adapter enters `execute_model` with a lazy
context provider and resolves spans only after vLLM has finalized its internal
`InputBatch` order. Both paths enable request-filtered traces and request-scoped
activation patches without changing other rows in a continuous batch.

Serving plugins start only on source-audited runtime package versions listed in the
compatibility matrix. `LDB_ALLOW_UNVERIFIED_RUNTIME=1` is an explicit development
override and must not be interpreted as compatibility evidence.

The control socket is created only after model topology discovery and normalized
hook installation complete. A process-local agent cannot be reused with a different
backend version, model identity, distributed world size, request-context contract,
or CUDA Graph mode.

A breakpoint stops the worker's current model thread, which pauses the whole forward
batch at that rank. It does not yet retract a single request while unrelated requests
continue. Breakpoints have mandatory bounded timeouts to avoid leaving a serving
worker stopped indefinitely after a client failure.

While stopped, use `inspect-hit HIT_ID` for the typed activation summary or
`inspect-hit HIT_ID --capture` to persist that exact activation as `safetensors`
before applying a patch and continuing. Inspection returns an empty hit on ranks
that do not own the requested breakpoint hit, so it remains safe through a
multi-rank client.

Full tensor capture synchronizes the selected tensor to CPU and writes it to disk.
Use narrow hook globs and isolated debug traffic. Summary capture also calls scalar
reductions and therefore synchronizes the device in the current implementation.
