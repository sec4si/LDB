# 权重导出、训练梯度与算子单步

这三个入口补充原来的激活 Debugger：`ldb weights` 修改并导出权重，
`ldb gradients` 观察真实 backward，`ldb debug-ops` 在可见 PyTorch 算子边界暂停。
它们面向独占的本地 eager 执行，不会自动挂接到 vLLM/SGLang 训练或推理 worker。
需要安装项目的 `transformers` 可选依赖。

## 永久修改权重

先查看实际参数名称，再选择修改对象。以下缩放只是操作示例，不是经过验证的模型修复：

```sh
ldb weights list openai-community/gpt2 --pattern 'transformer.h.0.mlp.*'
ldb weights export openai-community/gpt2 \
  --parameter transformer.h.0.mlp.c_proj.weight \
  --operation scale --value 0.99 --output runs/gpt2-edited
```

新目录包含模型配置、safetensors 权重分片、tokenizer 和 `ldb-weight-edit.json`。
模型重载后保留修改，不依赖 LDB 的激活补丁。原始模型目录不会被改写。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("runs/gpt2-edited", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("runs/gpt2-edited")
```

| 选项 | 含义 |
|---|---|
| `--operation scale --value 0.5` | 将选中的权重乘以 0.5；0 表示清零 |
| `--operation add --value 0.01` | 对选中的权重加标量 |
| `--operation replace --tensor replacement.safetensors` | 使用文件中的单个浮点张量替换；形状必须完全一致 |
| `--axis 1 --start 0 --stop 128` | 只改第 1 轴的 `[0:128]` 连续切片；索引非负、区间非空 |
| `--revision COMMIT` | 固定加载的模型 revision |
| `--max-shard-size 2GB` | 控制导出分片大小 |

Python 入口：`ldb.weight_editing.export_weight_edit(model, output, ...)`。
可传入 `tokenizer=` 保存 tokenizer；不传时仅导出模型与 manifest。

执行时保存原参数，在修改状态下调用 Transformers 的 `save_pretrained`，重新读取
导出的对应 safetensors 参数核对哈希，最后恢复内存中的参数数值。失败时移除本次
创建的输出目录，已有输出目录一律拒绝覆盖。manifest 记录来源、操作、切片、
tied aliases、参数修改前后哈希、替换文件哈希和全部导出文件哈希。

共享同一 Parameter 的 tied weights 会一起变化，并保留共享语义。例如修改 GPT-2
的 `lm_head.weight` 也会改变共享的输入 embedding，不能把它解释为只改输出头。
不同 Parameter 对象共享底层 storage 的 view 暂不支持。

边界：标准 Transformers 模型、eval、已物化的连续浮点权重；不支持量化模型、
PEFT 自动合并或 offload/meta 权重的通用转换。要求独占模型，不能在已有 autograd
graph 或并发训练过程中调用：参数数值会恢复，但写入仍改变 Tensor 的版本计数。
内存开销包括被编辑参数的 CPU 原值和修改副本；导出的是完整模型，需足够磁盘空间。
激活干预不会自动转化为权重更新，也没有实现 ROME/MEMIT 等知识编辑算法。

## 追踪真实训练梯度

下面执行两次 causal-LM forward/backward，并用 SGD 更新内存里的模型参数：

```sh
ldb gradients openai-community/gpt2 --device mps \
  --text 'When John and Mary went to the store, John gave a drink to Mary' \
  --parameter 'transformer.h.8.*' --module transformer.h.8.mlp \
  --steps 2 --learning-rate 0.0001 --capture --output runs/gpt2-gradients
```

`--parameter` 和 `--module` 是观察范围，不是训练参数的冻结规则。参数名与 module
名支持 glob，空匹配报错。默认 learning rate 为 0，只反传和观察，不执行 optimizer
更新；正值启用诊断用 SGD。每步先清空梯度。CLI 不把训练后的权重自动保存，
也不是完整 SFT/LoRA trainer。出现非有限 loss/gradient 时保存失败 trace 并拒绝
该次 optimizer 更新；此前已经完成的更新不会回滚。

输出 `trace.json` 包含以下事件，以及 loss、步号、更新次数、未使用/冻结参数状态。
加 `--capture` 时，原始张量保存在 `tensors.safetensors`。

| 事件 | 观察位置 |
|---|---|
| `incoming_gradient` | 本次反向传播到达参数的梯度，尚未代表最终累积值 |
| `accumulated_gradient` | 本次参数梯度累积完成后的 `.grad` |
| `activation` | 所选 module 的输出 |
| `activation_gradient` | 对该输出的梯度 |
| 自定义 snapshot 标签 | 调用时实际的 `.grad`，例如 unscale/clip 之后 |

统计包含 shape、dtype、device、L2 norm、最大绝对值、零值比例和非有限元素数。
NaN/Inf 触发 `nonfinite` 提示；有限张量的统计计算溢出单独标为
`statistic_nonfinite`，不混同于张量本身出现 NaN/Inf。`--norm-threshold` 只记录
阈值提示，不自动更改梯度。统计中的非有限数字以 JSON null 表示。

更重要的入口是接入已有训练循环，保留原来的 optimizer 和训练逻辑：

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

参数 hooks 不返回替换梯度，不改变模型 train/eval 状态，不执行 zero_grad、clip、
unscale 或 optimizer.step。梯度累积按现有训练循环进行；例如两次 backward 的
incoming 都是 8，累积事件可以分别是 8、16。`torch.autograd.grad` 不一定写入
`.grad`，不能据此要求出现累积事件。

AMP 用户应把实际 `GradScaler.get_scale()` 传给 `mark_step(loss_scale=...)`；
记录的是实际观测值，不会自动除以 scale。若在 `scaler.unscale_(optimizer)` 后
再取 snapshot，应先把声明的 loss_scale 改为 1。模型中嵌套开启的 autocast
仍由原训练代码管理。

trace 是单次、有限窗口；退出上下文（包括异常）会移除参数和激活 hooks。
默认最多 2,000 个事件、64 MiB 原始张量；超出预算显式计数，不把截断报告当作
完整记录。稀疏梯度只记录布局，不展开或宣称已检查其数值。统计和 CPU capture
会增加同步开销。DDP/FSDP/ZeRO 的全局梯度语义、跨 rank 汇总、optimizer/RNG
检查点恢复尚未覆盖。

## 单步执行可见算子

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

| 命令 | 行为 |
|---|---|
| `step` / `next` | 执行当前待执行算子，在下一个可见算子前暂停 |
| `continue` | 执行到下一个算子名称断点或函数完成 |
| `break PATTERN` / `delete PATTERN` | 添加/移除算子名称 glob，例如 `aten.mm.*` |
| `inspect` / `status` | 当前算子的输入结构、shape/dtype/device、完成计数和上一个输出结构 |
| `where` | 查看暂停线程的 Python 调用栈，帮助找到模型源码位置 |
| `history` | 最近完成的算子记录，默认保留 128 条元数据 |
| `capture PATH` | 保存当前算子的输入张量；目录需已存在，文件不覆盖 |
| `capture-output PATH` | 保存上一个已完成算子的当前输出张量 |
| `quit` | 请求取消，释放当前 Python 断点并等待工作线程结束 |

`--backward` 在同一次执行里计算 causal-LM loss 并 backward，因而也能暂停到
反向算子；模型保持 eval，目标是诊断梯度计算而非模拟带 dropout 的完整训练。
可以用 `-e step -e continue` 脚本化执行。

Python API 可以调试自己的模型、loss、optimizer 或自定义函数：

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

实现使用 `TorchDispatchMode`，保留一次真实执行的 Python/autograd 调用栈，在私有
工作线程里暂停；不是先跑完再播放日志。它继承创建时的 grad/inference 模式；
autocast、设备/stream 等其他线程局部上下文应放在传入函数内部建立。
默认在每个算子后同步涉及的 CUDA/MPS 设备，因此属于调试执行，不能用于测量正常吞吐。
单步包含该算子的所有底层工作，不保证一个算子只启动一个 GPU kernel。

**“算子”指 PyTorch dispatcher 可见的 ATen 或注册算子。** 自定义融合算子可以
作为整体暂停和执行，其内部 CUDA/Triton 指令不会展开。编译图、CUDA Graph replay、
其他线程发出的算子、opaque 第三方执行器不属于完整覆盖承诺。

默认最多执行 100,000 个可见算子，每次等待上限 30 秒。超时意味着仍在执行，
不会谎报停止；已进入的 C++/CUDA 调用或阻塞 Python 调用无法被线程级强制终止。
仅在后续 dispatcher 边界取消，close 等待仍超时则明确报错。
此模式不提供算子 undo、RNG/KV cache/optimizer 状态回滚。取消之前已经完成的
原地操作或训练更新保留；应使用独占的实验模型，重新开始实验时自行重置状态。
Python 属性赋值、控制流和 I/O 不属于算子单步，首次算子暂停前也可能已经执行。

## 验证范围

测试覆盖小型 GPT-2、Qwen3 的权重重载、tied weights、切片替换、BF16、分片和
异常恢复；直接训练与梯度追踪的梯度及 SGD 更新一致；真实 forward/backward
算子暂停、取消、错误和超时线程清理；以及 Mac MPS 路径。
这些证据不等于已经完成 70B 模型、多卡或线上服务的验收。

本次全量测试 183 项通过，无跳过；Ruff 和全部 39 个源码文件的 mypy 检查通过。
JUnit、梯度 trace 和小型测试模型的导出样例保存在
`runs/weights-gradients-operators-20260908/`。其中模型是测试配置，不是预训练模型的质量评估。

相关 API 依据：[Transformers 模型保存与加载](https://huggingface.co/docs/transformers/main_classes/model)、
[PyTorch 梯度 hooks](https://docs.pytorch.org/docs/2.14/notes/autograd.html)、
[PyTorch dispatcher 扩展](https://docs.pytorch.org/docs/2.14/notes/extending.html)。
