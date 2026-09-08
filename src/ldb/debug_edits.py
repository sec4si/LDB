"""Named, serializable activation edits for the offline layer debugger."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from ldb.models import FrozenModel
from ldb.sae import SAEDictionary, SAEFeature


class ActivationEdit(FrozenModel):
    kind: Literal["token", "head", "feature"]
    layer: int = Field(ge=0)
    scale: float = Field(allow_inf_nan=False)
    token: int | None = None
    head: int | None = Field(default=None, ge=0)
    dictionary: str | None = None
    dictionary_hash: str | None = None
    feature: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_selectors(self) -> Self:
        if (self.kind == "head") != (self.head is not None):
            raise ValueError("Only head edits require a head index")
        resources = (self.dictionary, self.dictionary_hash, self.feature)
        if self.kind == "feature" and any(value is None for value in resources):
            raise ValueError("Feature edits require a dictionary, hash and feature index")
        if self.kind != "feature" and any(value is not None for value in resources):
            raise ValueError("Dictionary selectors are only valid for feature edits")
        if self.kind == "token" and self.token is None:
            raise ValueError("Token edits require a token index")
        return self

    def validate_context(self, model: Any, hidden: Any) -> None:
        if self.layer >= len(model.transformer.h):
            raise ValueError("Edit layer is outside the model")
        if self.token is not None and not -hidden.shape[1] <= self.token < hidden.shape[1]:
            raise ValueError("Edit token is outside the input sequence")
        if self.head is not None:
            attention = model.transformer.h[self.layer].attn
            if attention.num_heads != model.config.n_head or getattr(attention, "pruned_heads", ()):
                raise ValueError("Head edits require an unpruned GPT-2 head layout")
            if self.head >= attention.num_heads:
                raise ValueError("Edit head is outside the attention layer")

    def load_feature(self, width: int) -> SAEFeature:
        assert self.dictionary is not None and self.feature is not None
        dictionary = SAEDictionary(Path(self.dictionary))
        if dictionary.digest != self.dictionary_hash:
            raise ValueError("SAE dictionary checksum mismatch")
        if dictionary.model_width != width:
            raise ValueError("SAE dictionary width does not match residual stream")
        return dictionary.feature(self.feature)

    def apply(self, tensor: Any, *, heads: int, feature: SAEFeature | None = None) -> Any:
        token = slice(None) if self.token is None else self.token
        channel = slice(None)
        if self.head is not None:
            width = tensor.shape[-1] // heads
            channel = slice(self.head * width, (self.head + 1) * width)
        result = tensor.clone()
        selected = tensor[:, token, channel]
        result[:, token, channel] = (
            feature.scale(selected, self.scale) if feature is not None else selected * self.scale
        )
        return result
