# Model Editing Capabilities and Training Research Value

LDB supports inference-time interventions, mechanism localization, and regression
verification, plus the newly added permanent weight export, training gradient tracing,
and stepping through visible PyTorch operators. The diagnostic CLI can run a limited
number of backward/SGD steps, but it is not a full LoRA/SFT trainer and has no
training-state checkpoint resume. For usage, see
[Weights, gradients, and operator debugging](weights-gradients-operators.md).
An effective activation intervention still does not mean "the model has learned it".

## What can be modified today

| Target | Supported operations | Entry point and scope | Permanently modifies the model? |
|---|---|---|---|
| Layer residual output | Scale, scale at specified tokens | GPT-2 layer debugger `patch`, `patch-token` | No; active for the current session |
| Attention head | Scale or zero before the output projection, optionally at specified tokens | GPT-2 layer debugger `patch-head`; existing offline/online head ablation paths | No; active for the current execution |
| SAE feature | Adjust the decoder contribution of a specified feature, optionally at specified tokens | GPT-2 layer debugger `patch-feature`; existing feature intervention API | No; active for the current execution |
| Other visible hook activations | scale, add, ablate, replace | Neural Patch / offline intervention API; depends on the hooks exposed by the adapter | No; inference-time patch |
| MoE routing | Add a bias to the router logit of a specified expert | Existing routing_bias intervention, on models that expose routing hooks | No; an input to the current routing selection |
| Visible expert outputs | Intervene on visible tensors | Available when the module is visible; cannot reach inside an opaque fused-MoE kernel | No |
| Weight tensors | Temporarily multiply by a finite scalar, restored on exit | Offline GPT-2 `edit_gpt2(parameter=..., scale=...)` | No; temporary change within the scope |
| Weight tensors of an exported model | scale, add, replace, optionally on a single-axis slice, preserving tied aliases | `ldb weights export`; standard non-quantized Transformers models | Yes, written to a new model directory; the original model's parameter values are restored |
| Layer execution path | Skip layers, repeat layers with shared weights | Offline GPT-2 `edit_gpt2(layer=..., passes=...)` | No; no independent parameters are added or removed |

The new interactive commands are still limited to GPT-2, eval, eager, and no KV cache.
The existing native Transformers, TransformerLens, vLLM, and SGLang adapters do not mean
that arbitrary model architectures have been validated either.
Online request-scoped control requires correct request/token alignment; the CUDA Graph
path only has experimental global scalar control.
See [runtime compatibility](compatibility.md) for detailed boundaries.

Persisting an activation checkpoint or a Neural Patch file only saves reusable debugging
state or intervention rules; it does not merge the rules into the weights. Permanent
weight changes use the separate `weights export` workflow.
`debug-ops` can pause at operator boundaries visible to the eager dispatcher, but cannot
expand the instructions inside a fused kernel. SAE dictionary training and permanent
layer/expert pruning are not implemented yet. The experiments directory has custom
o_proj LoRA training with merged export, but no general PEFT model management interface;
see the training value study (kept in a separate research repository; not published here).

## New interactive workflow

```bash
source .venv/bin/activate
ldb debug openai-community/gpt2 --device mps \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --prompt 'When John and Mary went to the store, John gave a drink to'
```

```text
tokens
target ' Mary' ' John'
snapshot original
compare "patch-head recipient 8 10 0 -1"
patch-head recipient 8 10 0 -1
edits
undo
redo
continue
scores
restore original
regress examples/suites/ioi-debugger.yaml "patch-head recipient 8 10 0 -1"
```

`compare` compares the baseline and a candidate edit from the same paused state, then
restores that state.
`regress` runs the current edit set and the candidate edit from scratch on new prompts,
and reports target-token scores and top-1 correct counts grouped by target, related, and
regression. Zeroing the head in this example is a research ablation, not a recommended fix.

Token and SAE edit examples:

```text
patch-token last-residual 8 -1 0.5
patch-feature feature-test 8 /path/to/layer8-sae.safetensors 12 0.5 -1
unpatch-edit last-residual
```

The SAE file must actually exist, follow the LDB dictionary format, and suit the layer
under study. Attaching semantic labels to feature indices requires sample analysis and
intervention evidence; the code being able to load a dictionary does not mean the
dictionary has validated semantics.
For exact commands, ordering, and checkpoint compatibility, see
[Offline layer debugger](layer-debugger.md).

## What value does this have for training

| Training stage | Evidence LDB can provide | How it informs training decisions | Work still to be done by external training/evaluation |
|---|---|---|---|
| Data preparation | Which input changes cause failures, which internal interventions change behavior | Collect matching counterexamples, rewrites, and controls to build a training set and an independent validation set | Labeling, sampling strategy, and actual training on the data |
| Choosing a fine-tuning approach | Intervention sensitivity of different layers/heads | Propose candidates for frozen layers, LoRA target modules, or local fine-tuning | Controlled training against standard approaches and random layers with the same data, steps, and budget |
| Analyzing the training process | Replay probes on checkpoints; GradientTracer records loss, parameter/activation gradients, and accumulated values | Compare when errors appear, locate non-finite values, observe changes in feature gradients | Large-scale training orchestration, distributed aggregation, optimizer state analysis and resume |
| Pruning and compression research | Effect of head ablation and layer skipping on behavior | Propose pruning candidates to be validated | Actually removing parameters, any necessary recovery training, speed/memory and quality measurement |
| Post-training regression | Score changes on target, related, and unrelated samples | Check whether target improvements come with regressions in other behavior | Broader generation evaluation, task benchmarks, and deployment acceptance |

The most direct value is **turning "pre-training guesses" into testable candidate
hypotheses, and turning failing samples into regression tests**.
Localization results do not produce gradients, nor do they automatically reduce training
loss or training cost.

For example, if a head has a clear causal contribution on a given task, one can compare:

1. A training setup with the standard LoRA target modules.
2. A setup that trains only the relevant modules in the candidate layers.
3. A random-layer control with the same parameter count and training budget.

All three use the same training set and are compared on a test set not used for
localization, selection, or training.
Only when target quality, generalization, regression, and measured training cost all
support it can one say that localization helped training.
LoRA trains a low-rank update to weight matrices, not an activation head directly;
see the [original LoRA paper](https://arxiv.org/abs/2106.09685).

Research has already shown cases where causal localization strength does not match where
knowledge editing succeeds, so "the location with the largest effect" cannot be directly
taken as "the best location to train/edit weights"; see [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213).

## Interpreting this project's existing GPT-2 results

In the earlier fixed IOI experiment, replacing head 8/10 recovered about 25.7% of the
original answer preference gap on the validation set, while zeroing it alone still left
every validation answer correct. This shows a partial causal contribution; it does not
prove the head can be safely removed, nor that fine-tuning only the parameters of that
head would learn the task well.

The earlier GPT-2 IOI study brought fine-grained editing and behavior comparison into the
interactive debugger, but did not validate any training benefit.
Training steps in the new gradient entry point are only used to verify the correctness of
observation and updates, and are not evidence of quality improvement.
The new Qwen3 training value study completed three equal-parameter-count LoRA training groups,
three seeds, permanent export, and offline reload verification. The localized group did
not outperform the standard or random groups; this round did not demonstrate a training
benefit. The task's clean donor itself had an insufficient success rate, so the next round
first needs to establish a reliable success/failure contrast on an independent calibration
set, then perform causal validation and training; this round's test set cannot be used for
tuning and still be claimed as unseen generalization afterwards.

## Verification for this change

- 126 tests pass, including consistency between head edits and direct hooks, token/SAE
  selection ranges, branch restore, exception cleanup, edit history, compatibility with
  old checkpoints, and refusal to load after a dictionary change.
- Ruff, mypy, and checkpoint JSON Schema consistency checks pass.
- On Mac MPS, the CLI `target`, `compare`, and `regress` were run against real GPT-2 at
  a pinned revision. The answer margin of the John/Mary example dropped from 2.52737 to
  1.48289, corresponding to zeroing head 8/10; after the 6 regression samples finished,
  the original session was still at layer 0.
- Raw CLI output and structured results are in `runs/mac-debugger-edits-20260908/`.
  SAE edit verification in this round used a synthetic dictionary and does not claim to
  validate real semantic features.

The follow-up training experiment after independent causal validation established a reliable
donor on a new Python dictionary task: replacing layer 19 raised independent validation
accuracy from 3.91% to 85.94%.
However, equal-budget local o_proj LoRA still did not pass the training benefit threshold.
Where an activation-input intervention is effective and which weights should be updated
need to be validated separately; whether high-scoring models retain single-step query
ability must also be included in regression.
