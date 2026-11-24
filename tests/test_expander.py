from __future__ import annotations

from contextlib import nullcontext

import pytest

import spatial_kis.expander as expander_module


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self, *, language_ids=None, vocab=None, decoded="a speedbolat finishes"):
        if language_ids is not None:
            self.lang_code_to_id = language_ids
        self._vocab = vocab or {}
        self.decoded = decoded
        self.src_lang = None

    def __call__(self, texts, **_kwargs):
        return FakeBatch(input_ids=texts)

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]

    def decode(self, _tokens, **_kwargs):
        return self.decoded


class FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


def install_factories(monkeypatch, tokenizer_factory, model_factory):
    tokenizer_calls = []
    model_calls = []
    monkeypatch.delenv("SPATIAL_KIS_TRANSLATION_MODEL", raising=False)

    def load_tokenizer(model_name, **kwargs):
        tokenizer_calls.append((model_name, kwargs))
        return tokenizer_factory(model_name, kwargs)

    def load_model(model_name, **kwargs):
        model_calls.append((model_name, kwargs))
        return model_factory(model_name, kwargs)

    monkeypatch.setattr(expander_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(expander_module.AutoModelForSeq2SeqLM, "from_pretrained", load_model)
    monkeypatch.setattr(expander_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(expander_module.torch, "autocast", lambda **_kwargs: nullcontext())
    return tokenizer_calls, model_calls


@pytest.mark.parametrize(
    ("tokenizer", "expected_decoder_id"),
    [
        (FakeTokenizer(language_ids={"en_XX": 42}), 42),
        (FakeTokenizer(vocab={"en_XX": 43}), 43),
    ],
)
def test_default_vinai_model_sets_languages_and_decoder_token(monkeypatch, tokenizer, expected_decoder_id):
    model = FakeModel()
    tokenizer_calls, _ = install_factories(
        monkeypatch,
        lambda _name, _kwargs: tokenizer,
        lambda _name, _kwargs: model,
    )

    service = expander_module.LocalExpander()
    translated = service.translate("về đích")

    assert service.model_name == "vinai/vinai-translate-vi2en-v2"
    assert tokenizer_calls[0] == (
        "vinai/vinai-translate-vi2en-v2",
        {"local_files_only": True, "src_lang": "vi_VN"},
    )
    assert tokenizer.src_lang == "vi_VN"
    assert model.generate_kwargs["decoder_start_token_id"] == expected_decoder_id
    assert model.generate_kwargs["num_beams"] == 3
    assert model.generate_kwargs["early_stopping"] is True
    assert model.generate_kwargs["max_new_tokens"] == 64
    assert model.generate_kwargs["do_sample"] is False
    assert translated == "a speedboat finishes"


def test_vinai_load_failure_falls_back_to_marian_without_decoder_token(monkeypatch, caplog):
    fallback_tokenizer = FakeTokenizer(decoded="a runner")
    fallback_model = FakeModel()

    def tokenizer_factory(model_name, _kwargs):
        return FakeTokenizer() if model_name != expander_module.FALLBACK_MODEL else fallback_tokenizer

    def model_factory(model_name, _kwargs):
        if model_name != expander_module.FALLBACK_MODEL:
            raise OSError("primary unavailable")
        return fallback_model

    tokenizer_calls, model_calls = install_factories(monkeypatch, tokenizer_factory, model_factory)

    service = expander_module.LocalExpander()
    assert service.translate("người chạy") == "a runner"

    assert service.model_name == expander_module.FALLBACK_MODEL
    assert [call[1]["local_files_only"] for call in model_calls] == [True, False, True]
    assert tokenizer_calls[-1] == (expander_module.FALLBACK_MODEL, {"local_files_only": True})
    assert "decoder_start_token_id" not in fallback_model.generate_kwargs
    assert "falling back" in caplog.text


def test_cleaning_cano_post_replacement_and_variants_deduplication(monkeypatch):
    cleaned, has_cano = expander_module.clean_vi_prompt("Có cảnh ca-nô về đích")
    assert cleaned == "thuyền cao tốc về đích"
    assert has_cano is True

    tokenizer = FakeTokenizer(decoded="speedbolat at sea.")
    model = FakeModel()
    install_factories(monkeypatch, lambda _name, _kwargs: tokenizer, lambda _name, _kwargs: model)
    service = expander_module.LocalExpander(expander_module.FALLBACK_MODEL)
    assert service.translate("ca-nô về đích") == "speedboat at sea"

    service.translate = lambda _text: "CANO VỀ ĐÍCH"
    monkeypatch.setattr(expander_module, "_EXPANDER_INSTANCE", service)
    assert expander_module.variants("cano về đích") == ["cano về đích"]
