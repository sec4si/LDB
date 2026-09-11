# MoE IR v1

An explicit intermediate representation between native checkpoint layouts,
layer-profile analysis and expert execution. It describes existing pretrained
expert bodies; it does not infer semantic Header/Body/Tail roles.

```text
Audited checkpoint + layer observations
                  ↓ import / shape and identity checks
                moe.ir/v1
  model identity · SSA graphs · tensor bindings · host boundary
                  · layer observations with provenance
                  ↓ validate / bind / lower
       CPU FP32 expert callable → existing host intervention
```

## Contract

- `graphs`: reusable, ordered SSA graphs with `linear`, `silu`, and `multiply`.
  Parameters use logical `[out, in]` shapes; experts bind graph parameters to
  named safetensors, explicit stored dtype and source SHA-256.
- `layers`: routing counts, graph reference and complete expert inventory.
- `boundary`: post-MoE-normalization input and unweighted expert output; routing
  weights and residual addition stay with the host. Compute dtype is explicit.
- `observations`: corpus-conditioned cluster/metrics plus hashed provenance.
  `semantic_role` remains unassigned; observations never prove compatibility.
- `sources`: payload/provenance references. JSON contains no tensors. Relative
  URIs resolve against the IR file directory in the reference runtime; explicit
  runtime overrides relocate weight files without changing their required hashes.

The backend-free validator rejects unsupported versions/operations, forward SSA
references, unknown operation fields, incompatible shapes, invalid routing counts,
duplicate/missing experts and incomplete weight bindings. Backend binding checks
actual source bytes, tensor shapes and storage dtype before execution.

`compatibility()` reports expert-body structure and whole-layer structure
separately. Matching expert graphs with different expert counts can be compatible
at the expert boundary while failing the whole-layer check. Functional
compatibility always remains `unverified`; empirical exchange evidence is separate.

## Commands

From the repository root, using its model-enabled environment:

```sh
python -m pytest tests/test_moe_ir.py
python -m ldb.experiments.moe_ir import-study --root runs/expert-abi-v1 --output runs/moe-ir-v1
python -m ldb.experiments.moe_ir validate runs/moe-ir-v1/model-0.json
python -m ldb.experiments.moe_ir inspect runs/moe-ir-v1/model-0.json
python -m ldb.experiments.moe_ir compare runs/moe-ir-v1/model-0.json runs/moe-ir-v1/model-2.json --left-layer 0 --right-layer 0
PYTHONPATH=. python scripts/verify_moe_ir.py --study runs/expert-abi-v1 --root runs/moe-ir-v1
```

New import destinations must be empty. This importer deliberately supports only
the audited native Mixtral SwiGLU assets from the existing study. Qwen packed
quantization, shared-expert branches, attention and full-model graphs are not
implemented. Frozen expert-ABI experiment sources remain unchanged.

```python
from ldb.experiments.moe_ir import load
from ldb.experiments.moe_ir.runtime import Runtime
from ldb.experiments.expert_abi.model import replace

ir = load('runs/moe-ir-v1/model-0.json')
runtime = Runtime(ir, base_dir='runs/moe-ir-v1')
function = runtime.expert(layer=3, expert=1)
# A trusted host model must correspond to this IR's identity and target slots.
with replace(model, [(3, 1, function)]):
    logits = model(**batch, use_cache=False).logits
```

The reference runtime loads the represented expert tensors once, converting to
CPU FP32. It is a correctness backend, not a lazy GPU runtime or optimized compiler.
Independent projections can use a different floating-point accumulation path
from fused native projections. The real-model verifier uses predeclared
`rtol=2e-4, atol=2e-4` for expert output and full-model logits, records maximum
errors, and requires bitwise baseline recovery after removing the intervention.

`bridge.link_package()` links existing `moe.expert-module/v0.1` packages to IR
host/donor slots, validating identities, payload hashes and packed-body dimensions.
H/T remains executed by the existing adapter backend, explicitly represented as
an external adapter in the link plan; v1 does not pretend to lower that adapter
into the expert SSA graph. Verification compares `T(IR_body(H(x)))` with the
existing package loader on captured native states for all six packages.
