# Reversible structural experiments

Run a short GPT-2 experiment on Apple MPS or CPU:

```console
ldb structural-test --device mps --output runs/mac-structural-test.json
```

The command loads `sshleifer/tiny-gpt2`, records baseline next-token logits, then
independently scales an existing MLP weight tensor to zero, skips a decoder block,
and executes that block twice using shared weights. After each experiment it
restores the original state and requires bit-exact baseline logits. Logit changes
demonstrate intervention effects, not improvements in model quality.

`--layer` chooses the block; `--model` is not an option: provide another GPT-2 model
as the positional model argument. `--prompt` selects the input.

The Python context manager exposes individual edits:

```python
from ldb.structural import edit_gpt2

with edit_gpt2(model, layer=0, passes=0):
    result = model(**inputs, use_cache=False)

with edit_gpt2(model, layer=0, passes=2):
    result = model(**inputs, use_cache=False)

with edit_gpt2(model, parameter="transformer.h.0.mlp.c_proj.weight", scale=0.5):
    result = model(**inputs, use_cache=False)
```

Use an exclusive, eager, eval-mode `GPT2LMHeadModel` with keyword inputs and no KV
cache. The context restores edits on exceptions as well as normal exit. Weight
snapshots consume CPU memory equal to the selected tensor. Weight changes affect
all aliases of a tied parameter while the context is active. Do not run concurrent
requests, training, compiled models, or generation with caches in this context.

This is an offline experimental API, not a Neural Patch format extension. It does
not add or delete parameters, layers, or experts, does not allocate independent
weights for repeated layers, and is not integrated with serving hot attachment.
