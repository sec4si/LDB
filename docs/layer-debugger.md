# Offline layer debugger

```console
ldb debug --device mps --prompt 'The Eiffel Tower is in'
```

This initial debugger supports eval-mode GPT-2 on CPU or Apple MPS. It executes
one complete prompt without a KV cache. It stops synchronously between decoder
blocks; no background inference thread or GPU kernel breakpoint is involved.

| Command | Behavior |
|---|---|
| `step` or `next` | Execute exactly one decoder block. |
| `continue` | Execute until a condition or watch fires, or the forward finishes. |
| `inspect` | Show the next layer, shape, norm, maximum magnitude and nonfinite count. |
| `tokens` | List input token positions, IDs, and decoded text for each batch row. |
| `break 0 norm 1` | Stop after layer 0 if its residual norm exceeds 1. |
| `break 0 nonfinite 0` | Stop after layer 0 if its output contains NaN or infinity. |
| `delete 0` | Remove layer 0's breakpoint. |
| `watch norm 0.1` | Stop when the residual norm changes by more than 0.1 in a step. |
| `unwatch norm` | Remove that watch. |
| `snapshot name` | Save the current activation, layer index and active patch map in memory. |
| `restore name` | Restore a checkpoint; `start` is saved automatically. |
| `snapshots` | List in-memory checkpoints and their next layer. |
| `save path.safetensors` | Save the current activation and patch map to a new file. |
| `load path.safetensors` | Validate and resume a disk checkpoint in this session. |
| `watch-feature recall 0 sae.safetensors 12 0.5` | Stop after layer 0 when SAE feature 12 exceeds 0.5. |
| `unwatch-feature recall` | Remove a named SAE watch. |
| `watches` or `breaks` | List active monitors or conditions. |
| `patch 1 0.5` | Scale layer 1's output by 0.5 when it next executes. |
| `patch-token last 1 -1 0.5` | Scale only the last token of layer 1's residual output. |
| `patch-head recipient 8 10 0 -1` | Zero head 10 before output projection at layer 8, final token only. |
| `patch-feature recall 1 sae.safetensors 12 0.5 -1` | Scale SAE feature 12 on the last residual token after layer 1. |
| `unpatch-edit recipient` | Remove a named edit affecting an unexecuted layer. |
| `edits` | List named edits, including token selectors and dictionary hashes. |
| `unpatch 1` | Disable that output patch. |
| `patches` | List active layer-output scales. |
| `undo` or `redo` | Navigate patch edits affecting unexecuted layers. |
| `branch 1 0` | Compare the current suffix with a branch that zeroes layer 1's output. |
| `target ' Mary' ' John'` | Select two single-token text answers; leading spaces matter. |
| `target-ids 5335 1757` | Select answer and alternative by token ID for the current tokenizer. |
| `scores` | After completion, show next-token text/IDs, target probabilities and logit differences. |
| `compare "patch-head recipient 8 10 0 -1"` | Compare baseline and edited suffix scores, then restore state/history. |
| `regress examples/suites/ioi-debugger.yaml "patch-head recipient 8 10 0 -1"` | Evaluate an edit on fresh target/related/regression prompts, preserving the paused session. |
| `logits` | Show the final next-token ID and maximum logit. |
| `quit` | Exit and release the model. |

Conditions use `norm`, `max_abs`, or `nonfinite`, and strict greater-than thresholds.
`step` always executes exactly one layer; `continue` honors post-layer conditions.
A branch runs both suffixes to completion regardless of breakpoints, reports the
maximum logit difference, and restores the paused activation and patch map.
Its baseline includes already-active patches. One output-scale patch exists per
layer; setting it again replaces its value and records undo history.

## Fine-grained edits and behavioral comparison

All layer, head, and token indices are zero-based; negative token indices count from the end of the input sequence.
When the final token argument is omitted, `patch-head` and `patch-feature` apply to all tokens.
These operations affect every row of the current batch; they do not select a single batch row.

```console
ldb debug openai-community/gpt2 --device mps \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --prompt 'When John and Mary went to the store, John gave a drink to' \
  -e "target ' Mary' ' John'" \
  -e 'compare "patch-head recipient 8 10 0 -1"' \
  -e 'regress examples/suites/ioi-debugger.yaml "patch-head recipient 8 10 0 -1"'
```

The `compare` baseline includes edits that are already active. The command accepts only a single patch/unpatch operation
and ignores breakpoint stop conditions while running the two suffix paths; on completion or error it restores the layer position, activation, edits, and undo/redo.
It returns, for each batch row, the answer probability, log probability, logit difference, and the mean change in the difference.
`preference_rate` is the fraction of rows in which the answer beats the specified competing token; it is not open-ended generation accuracy.
`scores` must be called after the last layer has executed. Each text target must be a single token and preserve the input tokenization boundary;
for multi-token answers, use the existing `ldb search` / `ldb patch verify`.

Named edits can coexist, and an operation with the same name replaces the existing edit; `undo`, `redo`, snapshots, and disk checkpoints all preserve them.
The execution order is fixed: head edits (before the output projection) → whole-layer residual scaling → token/SAE edits.
Edits of the same kind are applied in insertion order, and overlapping edits may compound; SAE interventions are generally not commutative.
Setting scale to 0 zeroes the selected activations or removes that SAE feature's decoder contribution; it does not delete model parameters.
Head edits currently reject pruned GPT-2 head layouts.

SAE edits reuse LDB's feature scaling: the reconstruction residual of the original activation is kept, and only the contribution of the specified decoder direction is adjusted.
This does not guarantee that the feature re-encodes to exactly the target value, nor that other features stay unchanged.
The dictionary is loaded when the edit is created, with its width, feature range, and file SHA-256 validated; which layer the dictionary corresponds to and its semantics must be verified by the researcher.

`regress` reuses `ldb.suite/v1alpha1` and currently accepts only a single next-token ID or a `target_text`
of exactly one token. It replays the current full edit set from the start of each fresh prompt and compares results before and after the extra command,
without reusing old prompt activations from before the pause point. Absolute token indices are interpreted per fresh prompt; out-of-range indices are an error.
It outputs per-example scores and, for each of the target/related/regression groups, the weighted mean log-probability change,
the worst change, and the top-1 correct count. It is an experiment report: it does not automatically issue a verified patch or make statistical significance judgments.
The example suite has only 6 examples and cannot represent broad capability regressions. The original paused state, breakpoints, and watch settings are not changed.

Example scripted session:

```console
ldb debug --device mps \
  -e 'break 0 norm 0' -e continue -e 'snapshot middle' \
  -e 'branch 1 0' -e 'patch 1 0' -e undo -e redo \
  -e continue -e 'restore middle' -e step -e logits
```

Named snapshots remain process-local. Disk checkpoints use a single safetensors
file with versioned metadata, the next layer, active output scales, and an activation
checksum. `save` refuses to overwrite an existing file. `load` checks exact model
weights, config, initial activation, attention mask, and other forward arguments
against the current session before changing its state. Reopen the same model with
the same prompt and input settings to resume. Weights are fingerprinted on session
creation and disk save/load, which incurs CPU transfer overhead on larger models.
Weights are not included in the checkpoint. Cross-device bit-exact replay is not
guaranteed; incompatible forward contexts fail validation.

New checkpoints use `ldb.layer-checkpoint/v1alpha2` and include named edits;
`v1alpha1` checkpoints without named edits remain readable. SAE dictionary tensors
are not embedded: metadata records their absolute paths and SHA-256 hashes. The
same files must be available on load; missing or changed dictionaries reject the
load before the current session is modified. Save revalidates dictionary files.
Targets, breakpoint/watch configuration, and undo/redo history remain session-local.

SAE watches use the supplied canonical LDB dictionary on the selected layer's
post-patch residual stream. They report the maximum feature activation across all
tokens and batch rows and stop on a strict threshold exceedance or a nonfinite
value. A dictionary's width is validated, but its semantic suitability for that
layer is the caller's responsibility. SAE encoding currently runs on CPU. Monitor
configuration is session-local and is not embedded in disk checkpoints.

Standard GPT-2 eval execution disables dropout; arbitrary stochastic
hooks, external weight edits, compiled execution, cross-attention, concurrent
requests, and training are outside this replay contract. Restoring clears undo/redo
history and preserves debugger breakpoint/watch settings.

This version watches residual statistics and SAE features, not expert routing.
It does not capture KV cache/RNG state, step individual operators, or extend the
vLLM/SGLang agent protocol. Zeroing layer output is distinct
from skipping the layer; use `structural-test` for the latter experiment.

Individual eager operator stepping is available separately through `ldb debug-ops`.
See [weights, gradients, and operator debugging](weights-gradients-operators.md) for
that execution boundary, permanent model export, and backward observations.
