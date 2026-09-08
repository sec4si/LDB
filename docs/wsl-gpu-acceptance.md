# WSL2 单卡验收记录（2026-09-08）

测试环境为 Ubuntu 24.04 / WSL2、RTX 4070 12GB、Windows NVIDIA 驱动
591.86。项目代码复制到独立目录，每个后端使用独立 Python 3.12 虚拟环境。
本次验证使用真实 HTTP completion 和 worker 内的 LDB agent。

模型为 `Qwen/Qwen3-0.6B`，固定 revision
`c1899de289a04d12100db370d81485cdf75e47ca`。范围为 prepared attachment、
单卡 eager、未量化权重，关闭前缀缓存；不代表热接入、CUDA Graph、TP/PP/DP
多卡、其他模型或训练流程通过验收。

## vLLM 0.28.0

**通过。** PyTorch `2.13.0+cu130`，FlashAttention 2，V1 model runner，
PyTorch sampler。对 `block.0.residual.post` 执行全局消融：

| 阶段 | 输出 token | token log probability |
|---|---|---:|
| 基线 | ` Paris` | -0.8785509467124939 |
| 消融 | `!` | -11.931214332580566 |
| 撤销 | ` Paris` | -0.8785509467124939 |

拥有该 hook 的 rank 0 产生原始 trace、含 patch ID 的 patched summary，
撤销后完整 `choices` 与基线一致。此强消融实验验证干预链路，不证明特定层的
语义专门化或局部知识编辑能力。

默认 V2 runner 在该 WSL2 环境报 `UVA is not available`。
V1 runner 默认 FlashInfer sampler 的 JIT 查找不到 CUDA toolkit，故这次明确
使用 PyTorch sampler。未修改 vLLM 源码。

```bash
source .venv-vllm/bin/activate
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
python scripts/gpu_smoke.py vllm Qwen/Qwen3-0.6B \
  --mode eager --port 18976 --artifacts "$PWD/evidence/vllm-qwen3-eager-fixed" -- \
  --host 127.0.0.1 --gpu-memory-utilization 0.45 \
  --max-model-len 1024 --max-num-seqs 4 --no-enable-prefix-caching \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

`HF_HUB_OFFLINE=1` 要求完整模型已缓存。端口必须空闲。

## SGLang 0.5.19

**通过。** PyTorch `2.13.0+cu130`，Triton attention，PyTorch sampler。
同一提示词和 hook 的结果为 ` Paris → ! → Paris`，基线与恢复的 token
log probability 均为 `-0.9114812612533569`，消融为 `-11.931214332580566`。
rank 0 的 trace、patch ID、patched summary 和完整 `choices` 恢复检查均通过。
两后端基线 log probability 不相同，本实验没有证明后端间数值等价。

安装固定版本需允许 SGLang 声明的预发布依赖：

```bash
uv venv --python 3.12 .venv-sglang
uv pip install --python .venv-sglang/bin/python --prerelease=allow \
  'sglang==0.5.19' -e .
source .venv-sglang/bin/activate
export CUDA_HOME="$PWD/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13"
```

本机使用 pip CUDA toolkit，没有 `/usr/local/cuda`。SGLang 的 JIT 链接器
要求 `$CUDA_HOME/lib64/libcudart.so`，在该独立环境中添加了两个原先缺失的
符号链接：`lib64 → lib`、`lib/libcudart.so → libcudart.so.13`。同时激活 venv，
使已安装的 `ninja` 可从 PATH 找到。未修改系统 CUDA 或 SGLang 源码。

```bash
HF_HUB_OFFLINE=1 python scripts/gpu_smoke.py sglang Qwen/Qwen3-0.6B \
  --mode eager --port 18977 --artifacts "$PWD/evidence/sglang-qwen3-eager-toolkit" -- \
  --host 127.0.0.1 --mem-fraction-static 0.45 --context-length 1024 \
  --max-running-requests 4 --disable-radix-cache \
  --attention-backend triton --sampling-backend pytorch \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

## 本次发现并修复的问题

- `ldb attach -e` 原先连续输出握手和命令结果两份 JSON，导致验收脚本
  `json.loads` 报 `Extra data`。现在仍执行握手，但仅交互模式打印握手内容。
- SGLang 未启用 DP 时 `dp_rank=None`，启动接入和热接入都需归一化为 rank 0。
  修复覆盖两条路径，保留非零 DP rank。

## 原始证据

本机目录：`runs/wsl-rtx4070-serving-20260908/`。
远端隔离目录：`/home/Kei/projects/ldb-gpu-acceptance-20260908/`。
证据包含服务日志、HTTP 响应、trace、patch summary、依赖冻结清单和源码哈希。
代码有未提交修改，报告中的 `ldb_commit=unknown` 不作为源码版本凭证；以
`source-manifest.json` 和实际执行文件的哈希为准。

两份成功报告位于 `evidence/vllm-qwen3-eager-fixed/result.json` 和
`evidence/sglang-qwen3-eager-toolkit/result.json`，对应的 `eager-evidence.json`
记录完整请求结果。失败尝试的日志也已保留。证据压缩包两端 SHA-256 一致：
`6d0c08c6353e4e84213cc5d5e62c64ea34248cef71e3befc241a00422ac44f84`。
测试结束后已确认两个测试端口释放，没有遗留测试服务进程。

安装方式参考 [vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
和 [SGLang 安装](https://docs.sglang.io/docs/get-started/install)。本记录的通过结论
来自上述机器的实际运行证据。
