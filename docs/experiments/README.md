# LDB experiment tools

Separate research repositories own the questions, protocols, and conclusions; LDB hosts the reusable tools. `ldb experiment list` lists the unified entry points, and `ldb experiment run NAME -- --help` shows a tool's arguments. All tools are launched through the current Python, and an installed wheel does not depend on where the research repositories are located.

| Source | Integrated capabilities | Limits |
| --- | --- | --- |
| Expert fingerprint research | Grouped SiLU fingerprints, response distance, routing-weighted replacement cost, random/frequency controls | Fixed coordinates, bias-free SiLU; response similarity depends on the corpus and does not prove interchangeability |
| MoE mixed-precision research | GPTQ, W4A16/W8A8 packing, fused activation pack, GU/down or separate gate/up/down allocation, restore, routing capture, PPL, timing | Qwen experiment adapters and an external MxMoE ABI; the allocator has 24-layer, 60-expert, 2048/1408-dimension constraints and is not an arbitrary MoE converter |
| MoE module research | Expert response capture, recoverable replacement, interface fitting, module packaging, MoE IR validate/compare/CPU execution, bounded budget search, quality gates, atomic delivery, packed expert sharing | Eager bank, Mixtral study, and Qwen-specific adapters; full budget execution is currently macOS-only, and MLX compression is Apple-only |

## Environment

Plain `ldb` and `experiment list` do not load Torch, NumPy, Triton, or MLX. CPU experiments need `numpy scipy torch transformers`, available through the `latent-debugger[experiments]` extra. Reuse an existing research environment directly and install only LDB itself (`pip install --no-deps -e /path/to/LDB`); do not reinstall CUDA PyTorch.

CUDA adapters additionally require an existing Triton and a matching **patched MxMoE** shared library, specified explicitly with each command's `--library`. LDB does not download weights, install CUDA, or compile kernels automatically. The library hash is part of the experiment freeze/restore checks. The original experiments' default-stream, FP16 input, and matrix-shape contracts still apply; new hardware must be re-validated, and old performance numbers are not a guarantee.

## Common entry points

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

`responses.npz` contains `outputs[expert, token, feature]` and `routes[token, expert]`, the latter being the actual weights after top-k. The output includes cosine, mean squared distance, routing-weighted error, and a chain-free donor mapping. A replacement plan still needs confirmation on an independent corpus.

Python API:

```python
from ldb.experiments.fingerprint import GroupedFingerprint
from ldb.experiments.expert_abi.model import replace
from ldb.experiments.expert_compaction.selection import response_costs, choose_map, allocate_layers
from ldb.experiments.moe_ir import load, compatibility
from ldb.experiments.moe_ir.runtime import Runtime
from ldb.experiments.precision.mxmoe_qwen_restore import restore
# CUDA packed Qwen expert sharing:
from ldb.experiments.sharing import SharedBank
```

See [the budget protocol](budget-builder.md) for the full budget spec, catalog, and split sample formats. See [the IR protocol](moe-ir.md) for the IR. Budget execution is rejected up front on Linux to avoid passing RSS off as macOS physical footprint. A quality gate failure does not force a model to be chosen, and a confirmation failure does not re-select using that data.

## Experiment discipline and migration notes

- Precision plans, sharing maps, and packed sizes do not equal end-to-end capability or inference speedup; record weight file bytes, peak memory, task scores, NLL, throughput, and latency separately.
- Use a new output directory, and first estimate the space needed for source weights, temporary packing, and final delivery. Budget delivery copies the model; this integration does not run that step.
- `src/ldb/experiments/provenance.json` records the repository, commit, and **working-tree file SHA256** for each migrated source file, including uncommitted experiment fixes; the original repositories were not modified.
- The migration adjusted module naming, working-directory resolution, and hook closure binding, and added unified entry points and regression tests. Research adapters keep dynamic typing; the new CLI is still subject to strict type checking.
- Not migrated: historical report generators, network preparation scripts, full-round orchestration scripts with hard-coded paths, model weights, and `.so` files. Same-family checkpoint deduplication/paging services and the full symbolic identity proof experiments remain in the original research repositories; this repository provides reusable fingerprint and intervention interfaces and does not treat every historical runner as a stable API.
- A traceable snapshot of the CUDA kernel source fixes is in `native/mxmoe-fixed4/`. It is a reproduction record of the original environment; the archived build.py.txt depends on the original MxMoE build tree and cannot be run as a portable install script.
