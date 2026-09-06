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

Scale a single feature without materializing the full latent vector. LDB evaluates
only the selected encoder column and applies its decoder-direction delta:

```console
(ldb) patch-set feature-18372 block.20.residual.post feature_scale 0.4 \
  --dictionary dictionaries/block-20.safetensors \
  --feature 18372 \
  --request request-42
```

The dictionary is loaded once when the patch is installed. Its selected vectors are
cached on the activation device and dtype after first use.
