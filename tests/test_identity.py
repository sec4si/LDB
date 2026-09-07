from types import SimpleNamespace

from ldb.identity import hash_tokenizer


class BackendTokenizer:
    def to_str(self) -> str:
        return '{"model":{"type":"BPE"}}'


def test_tokenizer_hash_is_deterministic_and_behavior_sensitive() -> None:
    first = SimpleNamespace(
        backend_tokenizer=BackendTokenizer(),
        special_tokens_map={"eos_token": "<eos>"},
        chat_template="{{ messages }}",
    )
    same = SimpleNamespace(
        backend_tokenizer=BackendTokenizer(),
        special_tokens_map={"eos_token": "<eos>"},
        chat_template="{{ messages }}",
    )
    changed = SimpleNamespace(
        backend_tokenizer=BackendTokenizer(),
        special_tokens_map={"eos_token": "<eos>"},
        chat_template="{{ messages[0] }}",
    )

    assert hash_tokenizer(first) == hash_tokenizer(same)
    assert hash_tokenizer(first) != hash_tokenizer(changed)
