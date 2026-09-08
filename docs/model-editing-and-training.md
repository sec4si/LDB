# 模型修改能力与训练研究价值

LDB 支持推理时干预、机制定位、回归验证，以及新增的永久权重导出、训练梯度追踪
和可见 PyTorch 算子单步。诊断 CLI 可执行有限 backward/SGD 步骤，但不是完整
LoRA/SFT 训练器，也没有训练状态检查点恢复。操作见
[权重、梯度和算子调试](weights-gradients-operators.md)。
一次激活干预有效仍不等于“模型已经学会了”。

## 当前可以修改什么

| 对象 | 支持的操作 | 入口和适用范围 | 是否永久修改模型 |
|---|---|---|---|
| 层 residual 输出 | 缩放、指定 token 缩放 | GPT-2 逐层 `patch`、`patch-token` | 否，当前会话激活 |
| attention head | 输出投影前缩放或清零，可指定 token | GPT-2 逐层 `patch-head`；已有离线/在线 head 消融路径 | 否，当前执行激活 |
| SAE 特征 | 调整指定 feature 的 decoder 贡献，可指定 token | GPT-2 逐层 `patch-feature`；已有 feature intervention API | 否，当前执行激活 |
| 其他可见 hook 激活 | scale、add、ablate、replace | Neural Patch / 离线干预 API；取决于 adapter 暴露的 hook | 否，推理时补丁 |
| MoE 路由 | 指定 expert 的 router logit 加偏置 | 已有 routing_bias 干预，在暴露路由 hook 的模型上使用 | 否，当前路由选择的输入 |
| 可见 expert 输出 | 对可见张量做干预 | 模块可见时可用；不支持穿透 opaque fused-MoE kernel | 否 |
| 权重张量 | 临时乘以一个有限标量，退出时恢复原值 | 离线 GPT-2 `edit_gpt2(parameter=..., scale=...)` | 否，作用域内临时改动 |
| 导出模型的权重张量 | scale、add、replace，可指定单轴切片，保留 tied aliases | `ldb weights export`；标准非量化 Transformers 模型 | 是，写入新模型目录；原模型参数数值恢复 |
| 层执行路径 | 跳过层、使用共享权重重复执行层 | 离线 GPT-2 `edit_gpt2(layer=..., passes=...)` | 否，不增删独立参数 |

新的交互命令仍限定 GPT-2、eval、eager、无 KV cache。已有原生 Transformers、
TransformerLens、vLLM、SGLang adapter 也不代表任意模型架构都已通过验证。
在线请求级控制要求能正确获得请求/token 对齐；CUDA Graph 路径只有实验性的全局标量控制。
详细边界见 [runtime compatibility](compatibility.md)。

持久保存一个激活检查点或 Neural Patch 文件，只是保存了可复用的调试状态或干预规则，
并没有把规则合并进权重。永久改权重使用独立的 `weights export` 流程。
`debug-ops` 可以在 eager dispatcher 可见的算子边界暂停，不能展开融合 kernel
内部指令。SAE 字典训练、层/专家永久剪枝尚未实现。实验目录已有定制 o_proj LoRA 训练与合并导出，
但不提供通用 PEFT 模型管理接口；见 [训练价值试验](../../expert-fingerprint-research/docs/training-value-study.md)。

## 新增的交互流程

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

`compare` 用同一个暂停现场比较基线和候选编辑，结束后恢复现场。
`regress` 在新 prompt 上从头执行当前编辑集合和候选编辑，按 target、related、regression
分组报告目标 token 分数和 top-1 正确数量。例子中的 head 清零是研究性消融，不是推荐的修复。

token 和 SAE 编辑示例：

```text
patch-token last-residual 8 -1 0.5
patch-feature feature-test 8 /path/to/layer8-sae.safetensors 12 0.5 -1
unpatch-edit last-residual
```

SAE 文件必须真实存在、符合 LDB 字典格式，且适合所研究的层。给 feature 编号贴语义标签之前，
需要样本分析与干预证据；代码能加载字典并不意味着该字典具有经过验证的语义。
具体命令、作用顺序和检查点兼容性见[逐层 Debugger](layer-debugger.md)。

## 对训练有什么价值

| 训练阶段 | LDB 能提供的证据 | 如何用于训练决策 | 尚需外部训练/评估完成的工作 |
|---|---|---|---|
| 数据准备 | 哪些输入变化导致失败、哪些内部干预改变行为 | 收集相应反例、改写和对照，构建训练集与独立验证集 | 标注、采样策略及实际数据训练 |
| 微调方案选择 | 不同层/head 的干预敏感性 | 提出冻结层、LoRA 目标模块或局部微调的候选方案 | 在相同数据、步数、预算下与常规方案及随机层对照训练 |
| 训练过程分析 | 在 checkpoint 上重放 probes；GradientTracer 记录 loss、参数/激活梯度及累积值 | 比较错误出现阶段、定位非有限值、观察特征梯度变化 | 大规模训练编排、分布式汇总、optimizer 状态分析与恢复 |
| 剪枝与压缩研究 | head 消融、层跳过对行为的影响 | 提出待验证的剪枝候选 | 真正删除参数、必要的恢复训练、速度/显存与质量测量 |
| 训练后回归 | 目标、相关、无关样本的分数变化 | 检查目标改善是否伴随其他行为退化 | 更广泛的生成评估、任务 benchmark 和部署验收 |

其中最直接的价值是**把“训练前的猜测”变成可检验的候选假设，并把失败样本沉淀为回归测试**。
定位结果不直接产生梯度，也不会自动降低训练 loss 或训练成本。

例如，若某个 head 在指定任务上有明显因果贡献，可以比较：

1. 常规 LoRA 目标模块的训练方案。
2. 只在候选层的相关模块上训练的方案。
3. 同等参数量和训练预算的随机层对照。

三者都使用同一训练集，并在未用于定位、选择或训练的测试集上比较。
只有目标质量、泛化、回归和实测训练成本一起支持，才能说定位帮助了训练。
LoRA 的训练对象是权重矩阵的低秩更新，并不是直接训练一个 activation head；
见 [LoRA 原论文](https://arxiv.org/abs/2106.09685)。

研究已经展示过因果定位强度与知识编辑成功位置不一致的情况，不能把“影响最大的位置”
直接认定为“最适合训练/编辑权重的位置”；见 [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213)。

## 对本项目现有 GPT-2 结果的解释

此前的固定 IOI 实验中，head 8/10 替换在验证集恢复了约 25.7% 的原答案偏好差，
单独清零后验证集仍全部答对。这说明它有部分因果贡献，不能证明它可以安全删除，
也不能证明只微调该 head 对应的参数就能学好任务。

此前 GPT-2 IOI 研究把精细编辑和行为对比接入交互 Debugger，未验证训练收益。
新增梯度入口的训练步骤仅用于验证观察与更新正确性，不作为质量提升证据。
新增 [Qwen3 训练价值试验](../../expert-fingerprint-research/docs/training-value-study.md) 已完成等参数量三组 LoRA 训练、
三个种子、永久导出和离线重载核验。定位组没有优于常规组或随机组；本轮没有证明训练收益。
任务的 clean donor 本身成功率不足，下一轮首先需要在独立 calibration 集建立可靠的
成功/失败对照，再做因果验证与训练；不能将本轮测试集用于调题后继续声称未见泛化。

## 本次验证

- 126 项测试通过，包括 head 编辑与直接 hook 的一致性、token/SAE 选择范围、
  分支恢复、异常清理、编辑历史、旧检查点兼容和字典变更拒绝加载。
- Ruff、mypy 与检查点 JSON Schema 一致性检查通过。
- Mac MPS 上使用固定 revision 的真实 GPT-2 执行了 CLI `target`、`compare`、
  `regress`。示例 John/Mary 的答案 margin 从 2.52737 降至 1.48289，
  对应 head 8/10 清零；6 个回归样本执行结束后，原会话仍在第 0 层。
- CLI 原始输出和结构化结果在 `runs/mac-debugger-edits-20260908/`。
  SAE 编辑的本次验证使用合成字典，不宣称验证了真实语义特征。

后续 [独立因果验证后的训练实验](../../expert-fingerprint-research/docs/validated-causal-training-study.md) 已在新的 Python 字典任务上
建立可靠 donor：第 19 层替换将独立验证准确率从 3.91% 提高到 85.94%。
但等预算局部 o_proj LoRA 仍未通过训练收益门槛。激活输入的有效干预位置与
应该更新哪些权重，需要分开验证；高分模型是否保留单步查询能力也必须纳入回归。
