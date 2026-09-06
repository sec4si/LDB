# SAE Dictionaries

LDB uses an explicit canonical `safetensors` layout instead of guessing external
library checkpoint conventions.

| Tensor | Shape | Required |
|---|---:|---:|
| `encoder` | `[model_width, feature_count]` | Yes |
| `decoder` | `[feature_count, model_width]` | Yes |
| `encoder_bias` | `[feature_count]` | No; defaults to zero |
| `decoder_bias` | `[model_width]` | No; defaults to zero |

Encoding is defined as:

```text
features = relu((activation - decoder_bias) @ encoder + encoder_bias)
```

Inspect the last captured tensor for a normalized hook:

```console
ldb features runs/<run-id> \
  --hook block.20.residual.post \
  --dictionary dictionaries/block-20.safetensors \
  --top-k 20 \
  --output feature-report.json
```

Reports bind the RunBundle ID, hook sequence, and SHA-256 of the exact dictionary.
