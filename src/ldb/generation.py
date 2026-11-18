"""Structured, single-request greedy generation for offline behavioral evaluation."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from ldb.models import FrozenModel


class ChatMessage(FrozenModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerationInput(FrozenModel):
    """Either raw text or fixed chat history, never an implicit chat-template guess."""

    prompt: str | None = Field(default=None, min_length=1)
    messages: tuple[ChatMessage, ...] = ()

    @model_validator(mode="after")
    def validate_input(self) -> GenerationInput:
        if (self.prompt is None) == (not self.messages):
            raise ValueError("Provide exactly one of prompt or non-empty messages")
        if self.messages and self.messages[-1].role != "user":
            raise ValueError("Chat history must end with a user message")
        return self


class GenerationSettings(FrozenModel):
    max_new_tokens: int = Field(default=128, ge=1, le=8192)
    do_sample: Literal[False] = False
    num_beams: Literal[1] = 1
    use_cache: bool = True


class GenerationResult(FrozenModel):
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...] = Field(min_length=1)
    generated_token_ids: tuple[int, ...]
    output_text: str
    raw_output_text: str
    # Raw payload excludes only an actual terminal EOS token, not other specials.
    raw_content_text: str
    finish_reason: Literal["eos", "length", "unknown"]
    chat_template_sha256: str | None = None
    # Snapshot supplied to generate(), before the library resolves defaults,
    # derives length limits, or constructs runtime cache/special-token objects.
    generation_config: dict[str, Any]
    generation_config_scope: Literal["explicit_request"] = "explicit_request"


def generate_transformers(
    model: Any,
    tokenizer: Any,
    device: str,
    request: GenerationInput,
    settings: GenerationSettings,
) -> GenerationResult:
    """Generate new tokens only, preserving the exact input and requested settings.

    This deliberately uses a fresh GenerationConfig: model-provided sampling,
    beam-search, forced-token, or stopping defaults must not silently affect a
    comparison. Only model/tokenizer BOS, EOS and padding identities are reused.
    """

    import torch
    from transformers import GenerationConfig

    if bool(getattr(model.config, "is_encoder_decoder", False)):
        raise ValueError("Generation evaluation requires a decoder-only model")
    if bool(getattr(model, "training", False)):
        raise ValueError("Generation evaluation requires model.eval()")

    template_hash = None
    if request.messages:
        template = tokenizer.get_chat_template()
        if not template:
            raise ValueError("Chat messages require a tokenizer chat template")
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        rendered = tokenizer.apply_chat_template(
            [message.model_dump() for message in request.messages],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
        )
        # The chat template already supplies its own special tokens.
        add_special_tokens = False
    else:
        assert request.prompt is not None
        rendered = request.prompt
        add_special_tokens = True
    if not isinstance(rendered, str):
        raise TypeError("Tokenizer returned a non-string rendered prompt")

    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        truncation=False,
    ).to(device)
    prompt_ids = tuple(int(token) for token in encoded["input_ids"][0])
    if not prompt_ids:
        raise ValueError("Generation input must contain at least one token")

    model_config = getattr(model, "generation_config", None)

    def special_token(name: str) -> Any:
        value = getattr(model_config, name, None)
        return value if value is not None else getattr(tokenizer, name, None)

    eos = special_token("eos_token_id")
    eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
    pad = special_token("pad_token_id")
    if pad is None and eos_ids:
        pad = eos_ids[0]
    config_factory = cast(Any, GenerationConfig)
    config = config_factory(
        **settings.model_dump(),
        bos_token_id=special_token("bos_token_id"),
        eos_token_id=eos,
        pad_token_id=pad,
        return_dict_in_generate=False,
    )
    # Recent Transformers fills unset fields from model.generation_config, even
    # when an explicit config is supplied. Isolate that source as well, restoring
    # it on every exit. This helper is only for serialized offline use.
    original_config = model.generation_config
    try:
        model.generation_config = config
        with torch.inference_mode():
            output = model.generate(**encoded, generation_config=config)
    finally:
        model.generation_config = original_config
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("Expected exactly one generated sequence")
    all_ids = tuple(int(token) for token in output[0])
    if all_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Generated sequence does not preserve the input token prefix")
    new_ids = all_ids[len(prompt_ids) :]
    if len(new_ids) > settings.max_new_tokens:
        raise ValueError("Generation exceeded max_new_tokens")
    reason: Literal["eos", "length", "unknown"] = "unknown"
    if new_ids and new_ids[-1] in eos_ids:
        reason = "eos"
    elif len(new_ids) == settings.max_new_tokens:
        reason = "length"
    return GenerationResult(
        rendered_prompt=rendered,
        prompt_token_ids=prompt_ids,
        generated_token_ids=new_ids,
        output_text=str(
            tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ),
        raw_output_text=str(
            tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        ),
        raw_content_text=str(
            tokenizer.decode(
                new_ids[:-1] if reason == "eos" else new_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        ),
        finish_reason=reason,
        chat_template_sha256=template_hash,
        generation_config=config.to_dict(),
    )
