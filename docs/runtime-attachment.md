# Runtime Attachment

## Attachment contract

LDB uses prepared attachment. The serving process loads a dormant runtime plugin at
startup, while the CLI may attach and detach later without restarting the model
server. LDB does not inject code into an arbitrary unprepared process.

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
agent directory. `ldb attach <directory>` discovers the live sockets and broadcasts
commands to every rank.

## Protocol

The current protocol version is `ldb.agent/v1alpha1`. Each request and response is
one bounded JSON object followed by a newline. Supported methods are:

| Method | Purpose |
|---|---|
| `hello` | Negotiate protocol, backend, topology, and capabilities. |
| `status` | Read arm state, event count, run ID, and active patches. |
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
| `continue` | Release one or all stopped model threads. |

`patch.set` supports scalar `scale`, `add`, and `ablate`; tensor-backed `add`
and `replace`; and an expert-specific `routing_bias`. Tensor files are loaded from
the serving worker's filesystem and must contain either one tensor or a tensor named
`activation`.

```console
(ldb) patch-set steer-1 'block.20.residual.post' add \
  --tensor /srv/ldb/steering.safetensors --request request-42
(ldb) patch-set route-1 'block.20.moe.router' routing_bias -0.25 \
  --expert 7 --request request-42
```

Unknown fields are rejected. Socket requests are limited to 1 MiB and responses to
16 MiB. Bulk tensors are written as `safetensors` instead of crossing the control
socket.

## Runtime integration

### SGLang

The `sglang.srt.plugins` entry point installs LDB after
`TpModelWorker.__init__`. It uses the loaded model's `named_modules()` topology and
registers normalized probes on decoder blocks, attention modules, MLP modules, and
visible MoE gates.

SGLang registers Python forward hooks after CUDA Graph capture. Those hooks execute
on eager forwards and do not execute during CUDA Graph replay. Operators must use
`--disable-cuda-graph` for complete live traces until LDB gains a graph-native probe
path.

### vLLM

The `vllm.general_plugins` entry point wraps the V1 GPU worker's model-load boundary
and installs the same normalized probes. The plugin is loaded only when both
`LDB_ENABLE=1` and `VLLM_PLUGINS=ldb` are present.

`--enforce-eager` is currently required for complete traces. The agent reports
`execution.eager_only` so clients cannot mistake a partial CUDA Graph trace for a
complete trace.

## Current safety boundary

The SGLang adapter propagates `ForwardBatch.rids` and prefill/decode token spans
through the model forward. The vLLM adapter enters `execute_model` with a lazy
context provider and resolves spans only after vLLM has finalized its internal
`InputBatch` order. Both paths enable request-filtered traces and request-scoped
activation patches without changing other rows in a continuous batch.

A breakpoint stops the worker's current model thread, which pauses the whole forward
batch at that rank. It does not yet retract a single request while unrelated requests
continue. Breakpoints have mandatory bounded timeouts to avoid leaving a serving
worker stopped indefinitely after a client failure.

Full tensor capture synchronizes the selected tensor to CPU and writes it to disk.
Use narrow hook globs and isolated debug traffic. Summary capture also calls scalar
reductions and therefore synchronizes the device in the current implementation.
