# Weight Export, Training Gradients, and Operator Stepping

These three entry points complement the existing activation debugger: `ldb weights`
modifies and exports weights, `ldb gradients` observes real backward passes, and
`ldb debug-ops` pauses at visible PyTorch operator boundaries.
They target exclusive, local eager execution and do not automatically attach to vLLM/SGLang
training or inference workers.
They require the project's optional `transformers` dependencies.

## Permanently modifying weights

First list the actual parameter names, then choose what to modify. The scaling below is
only an example of the operation, not a validated model fix:

```sh
ldb weights list openai-community/gpt2 --pattern 'transformer.h.0.mlp.*'
ldb weights export openai-community/gpt2 \
  --parameter transformer.h.0.mlp.c_proj.weight \
  --operation scale --value 0.99 --output runs/gpt2-edited
```

The new directory contains the model config, safetensors weight shards, the tokenizer,
and `ldb-weight-edit.json`. The modification persists after reloading the model and
does not depend on LDB activation patches. The original model directory is never rewritten.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("runs/gpt2-edited", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("runs/gpt2-edited")
```

| Option | Meaning |
|---|---|
| `--operation scale --value 0.5` | Multiply the selected weights by 0.5; 0 zeroes them |
| `--operation add --value 0.01` | Add a scalar to the selected weights |
| `--operation replace --tensor replacement.safetensors` | Replace with the single floating-point tensor in the file; shapes must match exactly |
| `--axis 1 --start 0 --stop 128` | Modify only the contiguous slice `[0:128]` along axis 1; indices must be non-negative and the range non-empty |
| `--revision COMMIT` | Pin the model revision to load |
| `--max-shard-size 2GB` | Control the exported shard size |

Python entry point: `ldb.weight_editing.export_weight_edit(model, output, ...)`.
Pass `tokenizer=` to save the tokenizer; without it, only the model and manifest are exported.

During execution, the original parameters are saved, Transformers' `save_pretrained` is
called in the modified state, the corresponding exported safetensors parameters are read
back to check hashes, and finally the in-memory parameter values are restored. On failure,
the output directory created by this run is removed; an existing output directory is always
refused rather than overwritten. The manifest records the source, operation, slice,
tied aliases, parameter hashes before and after the change, the replacement file hash, and
the hashes of all exported files.

Tied weights that share the same Parameter change together and keep their sharing
semantics. For example, modifying GPT-2's `lm_head.weight` also changes the shared input
embedding, and must not be interpreted as changing only the output head.
Views where different Parameter objects share underlying storage are not supported yet.

Boundaries: standard Transformers models, eval, materialized contiguous floating-point
weights; quantized models, automatic PEFT merging, and general conversion of offload/meta
weights are not supported. The model must be exclusive and must not be used inside an
existing autograd graph or a concurrent training process: parameter values are restored,
but the writes still change the Tensor's version counter.
Memory overhead includes CPU copies of the original values and modified copies of the
edited parameters; the full model is exported, so sufficient disk space is required.
Activation interventions are not automatically converted into weight updates, and
knowledge editing algorithms such as ROME/MEMIT are not implemented.

## Tracing real training gradients

The following runs two causal-LM forward/backward passes and updates the in-memory model
parameters with SGD:

```sh
ldb gradients openai-community/gpt2 --device mps \
  --text 'When John and Mary went to the store, John gave a drink to Mary' \
  --parameter 'transformer.h.8.*' --module transformer.h.8.mlp \
  --steps 2 --learning-rate 0.0001 --capture --output runs/gpt2-gradients
```

`--parameter` and `--module` define the observation scope, not freezing rules for trained
parameters. Parameter and module names support globs; an empty match is an error. The
default learning rate is 0, which only backpropagates and observes without an optimizer
update; a positive value enables diagnostic SGD. Gradients are cleared before each step.
The CLI does not save the trained weights automatically, and it is not a full SFT/LoRA
trainer. When a non-finite loss/gradient occurs, a failure trace is saved and that
optimizer update is refused; updates already completed before it are not rolled back.

The output `trace.json` contains the events below, plus the loss, step number, update
count, and unused/frozen parameter status.
With `--capture`, raw tensors are saved in `tensors.safetensors`.

| Event | Observation point |
|---|---|
| `incoming_gradient` | The gradient reaching the parameter in this backward pass; not yet the final accumulated value |
| `accumulated_gradient` | `.grad` after this parameter's gradient accumulation completes |
| `activation` | Output of the selected module |
| `activation_gradient` | Gradient with respect to that output |
| Custom snapshot label | The actual `.grad` at call time, e.g. after unscale/clip |

Statistics include shape, dtype, device, L2 norm, max absolute value, zero fraction, and
the number of non-finite elements.
NaN/Inf triggers a `nonfinite` flag; overflow while computing statistics on a finite tensor
is flagged separately as `statistic_nonfinite` and is not conflated with NaN/Inf in the
tensor itself. `--norm-threshold` only records a threshold flag and does not modify
gradients. Non-finite numbers in statistics are represented as JSON null.

The more important entry point integrates with an existing training loop, keeping the
original optimizer and training logic:

```python
from pathlib import Path
from ldb.gradients import GradientTracer

trace = GradientTracer(
    model, parameters=["model.layers.0.*"], modules=["model.layers.0.mlp"],
    capture=True, max_events=2000, max_tensor_bytes=64 * 1024 * 1024,
)
with trace:
    for step, batch in enumerate(batches):
        trace.mark_step(step, loss_scale=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        trace.record("loss", "loss", loss)
        loss.backward()
        trace.snapshot("before_optimizer")
        optimizer.step()
trace.save(Path("runs/training-window"))
```

Parameter hooks do not return replacement gradients, do not change the model's train/eval
state, and do not run zero_grad, clip, unscale, or optimizer.step. Gradient accumulation
follows the existing training loop; for example, if the incoming gradient of two backward
passes is 8 each time, the accumulation events can be 8 and 16. `torch.autograd.grad` does
not necessarily write to `.grad`, so accumulation events cannot be expected from it.

AMP users should pass the actual `GradScaler.get_scale()` to `mark_step(loss_scale=...)`;
recorded values are the actual observed values and are not automatically divided by the
scale. If a snapshot is taken after `scaler.unscale_(optimizer)`, the declared loss_scale
should first be changed to 1. Autocast enabled in nested form inside the model is still
managed by the original training code.

A trace is a single, bounded window; exiting the context (including via an exception)
removes the parameter and activation hooks.
By default it records at most 2,000 events and 64 MiB of raw tensors; overruns are counted
explicitly, and a truncated report is not presented as a complete record. Sparse gradients
only have their layout recorded; their values are not expanded or claimed to be checked.
Statistics and CPU capture add synchronization overhead. Global gradient semantics for
DDP/FSDP/ZeRO, cross-rank aggregation, and optimizer/RNG checkpoint resume are not
covered yet.

## Stepping through visible operators

```sh
ldb debug-ops openai-community/gpt2 --device mps \
  --prompt 'John gave a book to Mary'
```

```text
inspect
where
step
break aten.addmm.*
continue
capture runs/operator-input.safetensors
step
capture-output runs/operator-output.safetensors
delete aten.addmm.*
continue
```

| Command | Behavior |
|---|---|
| `step` / `next` | Execute the pending operator and pause before the next visible operator |
| `continue` | Run until the next operator-name breakpoint or until the function completes |
| `break PATTERN` / `delete PATTERN` | Add/remove an operator-name glob, e.g. `aten.mm.*` |
| `inspect` / `status` | Input structure of the current operator, shape/dtype/device, completed count, and the previous output structure |
| `where` | Show the Python call stack of the paused thread, to help locate the model source |
| `history` | Recently completed operator records; 128 metadata entries are kept by default |
| `capture PATH` | Save the current operator's input tensors; the directory must exist, files are not overwritten |
| `capture-output PATH` | Save the current output tensors of the last completed operator |
| `quit` | Request cancellation, release the current Python breakpoint, and wait for the worker thread to finish |

`--backward` computes the causal-LM loss and runs backward in the same execution, so it
can also pause at backward operators; the model stays in eval, since the goal is to
diagnose gradient computation rather than simulate full training with dropout.
Execution can be scripted with `-e step -e continue`.

The Python API can debug your own model, loss, optimizer, or custom functions:

```python
from ldb.operators import OperatorDebugger

def training_step():
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    return loss.detach()

with OperatorDebugger(training_step) as debugger:
    print(debugger.inspect())
    print(debugger.command("step"))
    print(debugger.command("continue"))
```

The implementation uses `TorchDispatchMode`, preserves the Python/autograd call stack of
a single real execution, and pauses in a private worker thread; it does not run to
completion first and then replay a log. It inherits the grad/inference mode at creation
time; other thread-local context such as autocast and device/stream should be set up
inside the function passed in.
By default it synchronizes the involved CUDA/MPS devices after each operator, so it is a
debugging execution and cannot be used to measure normal throughput.
A single step includes all the underlying work of that operator; one operator is not
guaranteed to launch only one GPU kernel.

**"Operator" means an ATen or registered operator visible to the PyTorch dispatcher.**
Custom fused operators can be paused and executed as a whole, but their internal
CUDA/Triton instructions are not expanded. Compiled graphs, CUDA Graph replay, operators
issued by other threads, and opaque third-party executors are not covered by any
full-coverage promise.

By default at most 100,000 visible operators are executed, and each wait is capped at
30 seconds. A timeout means execution is still in progress, and it does not falsely
report a stop; C++/CUDA calls already entered or blocking Python calls cannot be forcibly
terminated at the thread level.
Cancellation only happens at a subsequent dispatcher boundary; if waiting on close still
times out, an explicit error is raised.
This mode provides no operator undo and no rollback of RNG/KV cache/optimizer state.
In-place operations or training updates completed before cancellation are kept; use an
exclusive experimental model and reset state yourself when restarting an experiment.
Python attribute assignment, control flow, and I/O are not part of operator stepping, and
may already have executed before the first operator pause.

## Verification scope

Tests cover weight reloading, tied weights, slice replacement, BF16, sharding, and
exception recovery on small GPT-2 and Qwen3 models; agreement of gradients and SGD updates
between direct training and gradient tracing; pausing, cancellation, errors, and timed-out
thread cleanup for real forward/backward operators; and the Mac MPS path.
This evidence does not amount to acceptance on 70B models, multi-GPU, or online serving.

In this round, all 183 tests pass with none skipped; Ruff and mypy checks pass for all 39
source files.
JUnit results, gradient traces, and export samples of the small test models are saved in
`runs/weights-gradients-operators-20260908/`. The models there are test configurations,
not quality evaluations of pretrained models.

Related API references: [Transformers model saving and loading](https://huggingface.co/docs/transformers/main_classes/model),
[PyTorch gradient hooks](https://docs.pytorch.org/docs/2.14/notes/autograd.html),
[PyTorch dispatcher extension](https://docs.pytorch.org/docs/2.14/notes/extending.html).
