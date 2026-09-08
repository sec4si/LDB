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

## 精细编辑与行为对照

所有层、head、token 索引从 0 开始，负 token 索引从输入序列尾部计数。
`patch-head`、`patch-feature` 省略最后的 token 参数时作用于所有 token。
这些操作影响当前 batch 的所有行，不是选择某个 batch row。

```console
ldb debug openai-community/gpt2 --device mps \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --prompt 'When John and Mary went to the store, John gave a drink to' \
  -e "target ' Mary' ' John'" \
  -e 'compare "patch-head recipient 8 10 0 -1"' \
  -e 'regress examples/suites/ioi-debugger.yaml "patch-head recipient 8 10 0 -1"'
```

`compare` 的 baseline 包含当前已启用的编辑。命令只接受单个 patch/unpatch 操作，
执行两条后续路径时忽略断点的停止条件；结束或异常时恢复层位置、激活、编辑和 undo/redo。
它返回每个 batch row 的答案概率、log probability、logit 差，以及平均差值变化。
`preference_rate` 是答案胜过指定竞争 token 的比例，不是开放式生成准确率。
`scores` 必须在最后一层执行完后调用。文本 target 必须分别是单 token，并保持输入分词边界；
多 token 答案使用已有的 `ldb search` / `ldb patch verify`。

命名编辑可以共存，同名操作替换原编辑；`undo`、`redo`、快照和磁盘检查点均保存它们。
执行顺序固定为：head 编辑（输出投影前）→ 整层 residual 缩放 → token/SAE 编辑。
同类编辑按插入顺序应用，重叠的编辑可能叠加；SAE 干预通常不满足交换律。
将 scale 设为 0 表示清零选中激活或移除该 SAE feature 的 decoder 贡献，不表示删除模型参数。
head 编辑目前拒绝已裁剪的 GPT-2 head 布局。

SAE 编辑复用 LDB 的 feature scaling：保留原激活的重建残差，只调整指定 decoder 方向的贡献。
这不保证重新编码后该 feature 精确变成目标数值，也不保证其他 feature 不变。
字典在创建编辑时加载，校验宽度、feature 范围和文件 SHA-256；字典对应哪一层及其语义需研究者验证。

`regress` 复用 `ldb.suite/v1alpha1`，当前只接受单个 next-token ID 或恰好一个 token 的
`target_text`。它从每个新 prompt 的起点重放当前完整编辑集合，比较额外命令前后的结果，
不复用暂停点之前的旧 prompt 激活。绝对 token 索引按每个新 prompt 解释，越界直接报错。
输出逐样本分数，以及 target/related/regression 各组的加权平均 log-probability 变化、
最差变化和 top-1 正确数量。它是实验报告，不自动签发已验证补丁，也不作统计显著性判断。
示例 suite 只有 6 个样本，不能代表广泛能力回归。原暂停状态、断点和监视设置不会改变。

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
