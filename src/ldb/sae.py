"""Sparse autoencoder dictionary loading and feature inspection."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ldb.models import FrozenModel


class FeatureActivation(FrozenModel):
    feature: int = Field(ge=0)
    peak: float
    peak_token: int = Field(ge=0)
    mean: float
    active_fraction: float = Field(ge=0.0, le=1.0)


class TokenFeatureActivation(FrozenModel):
    feature: int = Field(ge=0)
    value: float


class TokenFeatureReport(FrozenModel):
    token: int = Field(ge=0)
    features: tuple[TokenFeatureActivation, ...]


class FeatureReport(FrozenModel):
    schema_version: Literal["ldb.features/v1alpha2"] = "ldb.features/v1alpha2"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    hook: str
    sequence: int = Field(ge=0)
    dictionary_hash: str
    token_count: int = Field(ge=0)
    feature_count: int = Field(gt=0)
    features: tuple[FeatureActivation, ...]
    tokens: tuple[TokenFeatureReport, ...] = ()


class SAEDictionary:
    """Canonical LDB SAE dictionary backed by safetensors."""

    def __init__(self, path: Path) -> None:
        import torch
        from safetensors.torch import load_file

        tensors = load_file(path)
        required = {"encoder", "decoder"}
        missing = required - tensors.keys()
        if missing:
            raise ValueError(f"SAE dictionary is missing tensors: {', '.join(sorted(missing))}")
        self.encoder = tensors["encoder"].float()
        self.decoder = tensors["decoder"].float()
        if self.encoder.ndim != 2 or self.decoder.ndim != 2:
            raise ValueError("SAE encoder and decoder must be matrices")
        if self.encoder.shape[1] != self.decoder.shape[0]:
            raise ValueError("SAE encoder and decoder feature dimensions do not match")
        if self.encoder.shape[0] != self.decoder.shape[1]:
            raise ValueError("SAE encoder and decoder model dimensions do not match")
        self.encoder_bias = tensors.get(
            "encoder_bias", torch.zeros(self.encoder.shape[1], dtype=torch.float32)
        ).float()
        self.decoder_bias = tensors.get(
            "decoder_bias", torch.zeros(self.encoder.shape[0], dtype=torch.float32)
        ).float()
        if self.encoder_bias.shape != (self.encoder.shape[1],):
            raise ValueError("SAE encoder_bias has an invalid shape")
        if self.decoder_bias.shape != (self.encoder.shape[0],):
            raise ValueError("SAE decoder_bias has an invalid shape")
        self.digest = _sha256_file(path)

    @property
    def model_width(self) -> int:
        return int(self.encoder.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.encoder.shape[1])

    def encode(self, activation: Any) -> Any:
        import torch

        if not isinstance(activation, torch.Tensor):
            raise TypeError("SAE inspection requires a torch.Tensor activation")
        if not activation.shape or activation.shape[-1] != self.model_width:
            raise ValueError(
                f"Activation width {activation.shape[-1:]} does not match SAE width "
                f"{self.model_width}"
            )
        values = activation.detach().float().reshape(-1, self.model_width).cpu()
        return torch.relu((values - self.decoder_bias) @ self.encoder + self.encoder_bias)

    def feature(self, index: int) -> SAEFeature:
        if index < 0 or index >= self.feature_count:
            raise ValueError(f"SAE feature {index} is outside dictionary size {self.feature_count}")
        return SAEFeature(
            encoder=self.encoder[:, index],
            decoder=self.decoder[index],
            encoder_bias=float(self.encoder_bias[index].item()),
            decoder_bias=self.decoder_bias,
        )


class SAEFeature:
    def __init__(
        self,
        *,
        encoder: Any,
        decoder: Any,
        encoder_bias: float,
        decoder_bias: Any,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_bias = encoder_bias
        self.decoder_bias = decoder_bias
        self._cache: dict[tuple[str, str], tuple[Any, Any, Any]] = {}

    def scale(self, activation: Any, factor: float) -> Any:
        import torch

        key = (str(activation.device), str(activation.dtype))
        values = self._cache.get(key)
        if values is None:
            values = tuple(
                tensor.to(device=activation.device, dtype=activation.dtype)
                for tensor in (self.encoder, self.decoder, self.decoder_bias)
            )
            self._cache[key] = values
        encoder, decoder, decoder_bias = values
        feature = torch.relu((activation - decoder_bias) @ encoder + self.encoder_bias)
        return activation + (factor - 1.0) * feature.unsqueeze(-1) * decoder


def inspect_sae_features(
    activation: Any,
    dictionary: SAEDictionary,
    *,
    top_k: int,
) -> tuple[FeatureActivation, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    encoded = dictionary.encode(activation)
    if encoded.shape[0] == 0:
        return ()
    peaks = encoded.amax(dim=0)
    means = encoded.mean(dim=0)
    active = (encoded > 0).float().mean(dim=0)
    count = min(top_k, dictionary.feature_count)
    indices = peaks.topk(count).indices.tolist()
    return tuple(
        FeatureActivation(
            feature=int(index),
            peak=float(peaks[index].item()),
            peak_token=int(encoded[:, index].argmax().item()),
            mean=float(means[index].item()),
            active_fraction=float(active[index].item()),
        )
        for index in indices
    )


def inspect_sae_tokens(
    activation: Any,
    dictionary: SAEDictionary,
    *,
    top_k: int,
) -> tuple[TokenFeatureReport, ...]:
    """Return each flattened token's strongest SAE features."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    encoded = dictionary.encode(activation)
    count = min(top_k, dictionary.feature_count)
    reports = []
    for token, row in enumerate(encoded):
        indices = row.topk(count).indices.tolist()
        reports.append(
            TokenFeatureReport(
                token=token,
                features=tuple(
                    TokenFeatureActivation(feature=int(index), value=float(row[index].item()))
                    for index in indices
                ),
            )
        )
    return tuple(reports)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
