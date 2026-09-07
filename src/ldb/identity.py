"""Canonical model and tokenizer identity for reproducible runtime attachment."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from ldb.models import TokenizerIdentity


def hash_tokenizer(tokenizer: Any) -> str:
    """Hash the tokenizer graph and behavior-affecting configuration."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None else None
    payload = json.dumps(
        {
            "backend": backend_json,
            "special_tokens_map": tokenizer.special_tokens_map,
            "chat_template": getattr(tokenizer, "chat_template", None),
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_tokenizer_identity(
    model_id: str, *, revision: str | None = None
) -> TokenizerIdentity:
    """Resolve an HF revision and hash the tokenizer loaded from that exact commit."""

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    resolved = HfApi().model_info(model_id, revision=revision).sha
    if not resolved:
        raise ValueError(f"Hugging Face did not resolve an immutable revision for {model_id!r}")
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_id, revision=resolved)
    return TokenizerIdentity(
        model_id=model_id,
        model_revision=resolved,
        tokenizer_hash=hash_tokenizer(tokenizer),
    )
