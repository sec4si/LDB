# SAE Dictionaries

LDB uses an explicit canonical `safetensors` layout instead of guessing external
library checkpoint conventions.

| Tensor | Shape | Required |
|---|---:|---:|
| `encoder` | `[model_width, feature_count]` | Yes |
| `decoder` | `[feature_count, model_width]` | Yes |
| `encoder_bias` | `[feature_count]` | No; defaults to zero |
| `decoder_bias` | `[model_width]` | No; defaults to zero |
| `input_bias` | `[model_width]` | No; defaults to `decoder_bias` |

Encoding and decoding are defined as:

```text
features = relu((activation - input_bias) @ encoder + encoder_bias)
reconstruction = features @ decoder + decoder_bias
```

An SAE reconstructs the activation it encodes, so `input_bias` is normally omitted and
the encoder centers on `decoder_bias`.

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

## Transcoders

A transcoder uses the same layout but encodes an MLP's input and reconstructs that MLP's
output, standing in for the MLP's computation. Give it its own `input_bias` when the
input centering differs from the output bias. Capture both sides in one run and compare:

```console
ldb run <model> --prompt "..." --capture 'block.5.mlp.*'
ldb transcoder runs/<run-id> \
  --input-hook block.5.mlp.input \
  --output-hook block.5.mlp.output \
  --dictionary dictionaries/block-5-transcoder.safetensors \
  --output transcoder-report.json
```

The report lists the strongest features on the MLP input and `normalized_mse`: the
squared error between the transcoder's reconstruction and the real MLP output, divided by
that output's variance. `0` is a perfect stand-in; values near or above `1` explain
nothing. A transcoder that reproduces the output can still do so through a different
mechanism than the MLP, so treat its features as candidates to verify with interventions.
