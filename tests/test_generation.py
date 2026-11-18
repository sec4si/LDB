"""Contract tests use real tokenizers/configs and a minimal tensor/model double.

These do not count as model execution. A separate torch-gated integration test
exercises a tiny locally constructed model when the platform supports PyTorch.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ldb.generation import ChatMessage, GenerationInput, GenerationSettings, generate_transformers

transformers = pytest.importorskip("transformers")


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.ndim = 2
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        return self.rows[index]


class Encoding(dict):
    def to(self, device):
        assert device == "cpu"
        return self


class Tokenizer:
    eos_token_id = 9
    bos_token_id = 1
    pad_token_id = None

    def __init__(self):
        self.calls = []
        self.template = "test-template"

    def get_chat_template(self):
        return self.template

    def apply_chat_template(self, messages, **kwargs):
        self.chat_args = (messages, kwargs)
        return "<bos>SECRET<assistant>"

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return Encoding(input_ids=Tensor([[1, 2]]), attention_mask=Tensor([[1, 1]]))

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not clean_up_tokenization_spaces
        mapping = {1: "<bos>", 2: "SECRET", 3: " answer", 9: "<eos>"}
        return "".join(mapping[token] for token in ids if not (skip_special_tokens and token == 9))


class Model:
    config = SimpleNamespace(is_encoder_decoder=False)
    training = False

    def __init__(self, tokens=(3, 9)):
        self.generation_config = transformers.GenerationConfig(
            eos_token_id=[9, 8], do_sample=True, stop_strings=["unsafe inherited stop"]
        )
        self.original = self.generation_config
        self.tokens = tokens
        self.fail = False

    def generate(self, **kwargs):
        self.kwargs = kwargs
        config = kwargs["generation_config"]
        assert self.generation_config is config
        assert config.do_sample is False
        assert config.stop_strings is None
        if self.fail:
            raise RuntimeError("inference error")
        return Tensor([[1, 2, *self.tokens]])


@pytest.fixture
def fake_torch(monkeypatch):
    # Real transformers config is imported first; only the helper's torch context
    # is replaced. Model/token tensors in this file are explicitly test doubles.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))


def test_generated_tokens_are_sliced_before_decode_and_settings_restored(fake_torch):
    model, tokenizer = Model(), Tokenizer()
    generated = generate_transformers(
        model,
        tokenizer,
        "cpu",
        GenerationInput(prompt="SECRET"),
        GenerationSettings(max_new_tokens=2),
    )
    assert generated.output_text == " answer"
    assert generated.raw_output_text == " answer<eos>"
    assert generated.raw_content_text == " answer"
    assert generated.generation_config_scope == "explicit_request"
    assert generated.prompt_token_ids == (1, 2)
    assert generated.generated_token_ids == (3, 9)
    assert generated.finish_reason == "eos"  # EOS on last permitted token is complete.
    assert generated.chat_template_sha256 is None
    assert model.generation_config is model.original
    assert generated.generation_config["do_sample"] is False
    assert tokenizer.calls[0][1]["add_special_tokens"] is True
    assert tokenizer.calls[0][1]["truncation"] is False


def test_chat_template_is_explicit_and_special_tokens_not_duplicated(fake_torch):
    model, tokenizer = Model(), Tokenizer()
    request = GenerationInput(messages=(ChatMessage(role="user", content="hello"),))
    generated = generate_transformers(model, tokenizer, "cpu", request, GenerationSettings())
    assert tokenizer.calls[0][1]["add_special_tokens"] is False
    assert tokenizer.chat_args[1]["add_generation_prompt"] is True
    assert tokenizer.chat_args[1]["chat_template"] == "test-template"
    assert len(generated.chat_template_sha256) == 64
    tokenizer.template = None
    with pytest.raises(ValueError, match="chat template"):
        generate_transformers(model, tokenizer, "cpu", request, GenerationSettings())


@pytest.mark.parametrize(
    "tokens,expected", [((3, 3), "length"), ((3,), "unknown"), ((), "unknown")]
)
def test_finish_reasons(fake_torch, tokens, expected):
    generated = generate_transformers(
        Model(tokens),
        Tokenizer(),
        "cpu",
        GenerationInput(prompt="x"),
        GenerationSettings(max_new_tokens=2),
    )
    assert generated.finish_reason == expected


def test_generation_exception_restores_original_config(fake_torch):
    model = Model()
    model.fail = True
    with pytest.raises(RuntimeError, match="inference error"):
        generate_transformers(
            model, Tokenizer(), "cpu", GenerationInput(prompt="x"), GenerationSettings()
        )
    assert model.generation_config is model.original


def test_unexpected_prefix_is_rejected_and_config_restored(fake_torch):
    model = Model()
    model.generate = lambda **kwargs: Tensor([[2, 1, 3, 9]])
    with pytest.raises(ValueError, match="prefix"):
        generate_transformers(
            model, Tokenizer(), "cpu", GenerationInput(prompt="x"), GenerationSettings()
        )
    assert model.generation_config is model.original


def test_training_and_encoder_decoder_are_rejected(fake_torch):
    model = Model()
    model.training = True
    with pytest.raises(ValueError, match="eval"):
        generate_transformers(
            model, Tokenizer(), "cpu", GenerationInput(prompt="x"), GenerationSettings()
        )
    model.training = False
    model.config = SimpleNamespace(is_encoder_decoder=True)
    with pytest.raises(ValueError, match="decoder-only"):
        generate_transformers(
            model, Tokenizer(), "cpu", GenerationInput(prompt="x"), GenerationSettings()
        )


def test_tiny_real_model_generation_without_downloads():
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    backend = BackendTokenizer(WordLevel({"<eos>": 0, "hello": 1, "<unk>": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=3,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=16,
            bos_token_id=0,
            eos_token_id=0,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    output = generate_transformers(
        model,
        tokenizer,
        "cpu",
        GenerationInput(prompt="hello"),
        GenerationSettings(max_new_tokens=3),
    )
    assert output.prompt_token_ids == (1,)
    assert output.generated_token_ids == (0,)
    assert output.finish_reason == "eos"
    assert output.output_text == ""
