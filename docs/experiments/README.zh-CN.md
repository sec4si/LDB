# LDB 实验工具

三个研究仓库负责问题、协议和结论；LDB 承载可复用工具。`ldb experiment list` 列出统一入口，`ldb experiment run NAME -- --help` 查看该工具参数。所有工具通过当前 Python 启动，安装后的 wheel 不依赖三个研究仓库的位置。

| 来源 | 已集成能力 | 边界 |
| --- | --- | --- |
| expert-fingerprint-research | 分组 SiLU 指纹、响应距离、路由加权替换代价、随机/频率对照 | 固定坐标、无偏置 SiLU；响应相似度依赖语料，不证明可互换 |
| moe-mixed-precision-research | GPTQ、W4A16/W8A8 打包、融合 activation pack、GU/down 或独立 gate/up/down 分配、恢复、路由捕获、PPL、计时 | Qwen 实验适配器和外部 MxMoE ABI；分配器含 24 层、60 专家、2048/1408 维度约束，不是任意 MoE 转换器 |
| moe-module-research | 专家响应捕获、可恢复替换、接口拟合、模块封装、MoE IR 校验/比较/CPU 执行、有限预算搜索、质量门槛、原子交付、packed 专家共享 | eager bank、Mixtral study 和 Qwen 专用适配器；完整预算执行目前仅 macOS，MLX 压缩仅 Apple 平台 |

## 环境

普通 `ldb` 和 `experiment list` 不加载 Torch、NumPy、Triton 或 MLX。CPU 实验需 `numpy scipy torch transformers`，可使用 `latent-debugger[experiments]` extra。已有研究环境请直接复用，仅安装 LDB 本身（`pip install --no-deps -e /path/to/LDB`），不要重复安装 CUDA PyTorch。

CUDA 适配器另外要求现有 Triton 和匹配的 **patched MxMoE** 动态库，使用每个命令的 `--library` 明确指定。LDB 不下载权重、不安装 CUDA、不自动编译内核。库哈希会参与实验冻结/恢复检查。原实验的 default-stream、FP16 输入和矩阵形状契约仍然有效；新硬件必须重新验收，不能把旧性能数字作为保证。

## 常用入口

```bash
ldb experiment list
ldb experiment run ir -- validate model-ir.json
ldb experiment run ir -- compare host.json donor.json --left-layer 0 --right-layer 0
ldb experiment run fingerprint -- --input responses.npz --output comparison.json --remove 1
ldb experiment run precision-build -- --help
ldb experiment run precision-profile -- --help
ldb experiment run precision-plan -- --help
ldb experiment run precision-linear-plan -- --help
ldb experiment run precision-ppl -- --help
ldb experiment run budget -- --catalog catalog.json --spec spec.json \
  --development development.json --confirmation confirmation.json \
  --output new-run --plan-only
```

`responses.npz` 包含 `outputs[expert, token, feature]` 和 `routes[token, expert]`，后者为 top-k 后实际权重。输出含 cosine、均方距离、路由加权误差和无链 donor mapping。替换计划仍需独立语料确认。

Python API：

```python
from ldb.experiments.fingerprint import GroupedFingerprint
from ldb.experiments.expert_abi.model import replace
from ldb.experiments.expert_compaction.selection import response_costs, choose_map, allocate_layers
from ldb.experiments.moe_ir import load, compatibility
from ldb.experiments.moe_ir.runtime import Runtime
from ldb.experiments.precision.mxmoe_qwen_restore import restore
# CUDA packed Qwen 专家共享：
from ldb.experiments.sharing import SharedBank
```

完整预算规格、catalog 和拆分样本格式见 [预算协议](budget-builder.md)。IR 见 [IR 协议](moe-ir.md)。预算执行在 Linux 上提前拒绝，避免拿 RSS 冒充 macOS physical footprint。质量门槛失败不会强制选一个模型，confirmation 失败不会用该数据重新挑选。

## 实验纪律与迁移记录

- 精度计划、共享映射和打包大小不等于端到端能力或推理加速；分别记录权重文件字节数、峰值内存、任务分数、NLL、吞吐和延迟。
- 使用新输出目录，先估算源权重、临时打包和最终交付所需空间。预算交付会复制模型；本次集成不运行该步骤。
- `src/ldb/experiments/provenance.json` 记录每份迁入源码的仓库、提交和 **工作区文件 SHA256**，包括未提交的实验修复；原仓库未被修改。
- 迁移时调整了模块命名、工作目录解析和 hook 闭包绑定，添加统一入口与回归测试。研究适配器保留动态类型；新的 CLI 仍接受严格类型检查。
- 未迁入历史报告生成器、联网准备脚本、硬编码路径的整轮调度脚本、模型权重和 `.so`。同族 checkpoint 去重/分页服务和完整符号恒等证明实验仍保留原研究仓库；此处提供可复用指纹与干预接口，不把所有历史 runner 当作稳定 API。
- CUDA 内核源修复的可追溯快照见 `native/mxmoe-fixed4/`。它是原环境的复现记录，归档的 build.py.txt 依赖原 MxMoE 构建树，不能作为可移植安装脚本运行。
