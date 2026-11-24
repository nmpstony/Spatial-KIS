from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

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


class FakeModuleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.generation_config = SimpleNamespace(max_length=20)
        self.generate_kwargs = None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


def install_factories(monkeypatch, tokenizer_factory, model_factory, *, cuda=False):
    tokenizer_calls = []
    model_calls = []
    monkeypatch.delenv("SPATIAL_KIS_TRANSLATION_MODEL", raising=False)
    monkeypatch.delenv("SPATIAL_KIS_QUANTIZE_TRANSLATOR", raising=False)

    def load_tokenizer(model_name, **kwargs):
        tokenizer_calls.append((model_name, kwargs))
        return tokenizer_factory(model_name, kwargs)

    def load_model(model_name, **kwargs):
        model_calls.append((model_name, kwargs))
        return model_factory(model_name, kwargs)

    monkeypatch.setattr(expander_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(expander_module.AutoModelForSeq2SeqLM, "from_pretrained", load_model)
    monkeypatch.setattr(expander_module.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(expander_module.torch, "autocast", lambda **_kwargs: nullcontext())
    return tokenizer_calls, model_calls


def test_cpu_default_is_marian_greedy_and_unquantized(monkeypatch):
    tokenizer = FakeTokenizer(decoded="a runner")
    model = FakeModuleModel()
    tokenizer_calls, _ = install_factories(
        monkeypatch,
        lambda _name, _kwargs: tokenizer,
        lambda _name, _kwargs: model,
    )
    monkeypatch.setattr(
        expander_module.torch.ao.quantization,
        "quantize_dynamic",
        lambda *_args, **_kwargs: pytest.fail("quantization must be opt-in"),
    )

    service = expander_module.LocalExpander()
    service.translate("ng\u01b0\u1eddi ch\u1ea1y")

    assert service.device.type == "cpu"
    assert service.model_name == expander_module.CPU_DEFAULT_MODEL == expander_module.FALLBACK_MODEL
    assert tokenizer_calls[0] == (
        expander_module.FALLBACK_MODEL,
        {"local_files_only": True},
    )
    assert model.generate_kwargs["num_beams"] == 1
    assert "early_stopping" not in model.generate_kwargs
    assert "decoder_start_token_id" not in model.generate_kwargs


@pytest.mark.parametrize(
    ("cuda", "model_selection"),
    [(True, "default"), (False, "constructor"), (False, "environment")],
)
@pytest.mark.parametrize(
    ("tokenizer", "expected_decoder_id"),
    [
        (FakeTokenizer(language_ids={"en_XX": 42}), 42),
        (FakeTokenizer(vocab={"en_XX": 43}), 43),
    ],
)
def test_vinai_selection_sets_languages_decoder_token_and_device_decoding(
    monkeypatch, cuda, model_selection, tokenizer, expected_decoder_id
):
    model = FakeModel()
    tokenizer_calls, _ = install_factories(
        monkeypatch,
        lambda _name, _kwargs: tokenizer,
        lambda _name, _kwargs: model,
        cuda=cuda,
    )
    if model_selection == "environment":
        monkeypatch.setenv("SPATIAL_KIS_TRANSLATION_MODEL", expander_module.DEFAULT_MODEL)
    constructor_model = expander_module.DEFAULT_MODEL if model_selection == "constructor" else None
    monkeypatch.setattr(
        expander_module.torch.ao.quantization,
        "quantize_dynamic",
        lambda *_args, **_kwargs: pytest.fail("CUDA model should not be quantized"),
    )

    service = expander_module.LocalExpander(constructor_model)
    translated = service.translate("v\u1ec1 \u0111\u00edch")

    assert service.model_name == expander_module.DEFAULT_MODEL
    assert tokenizer_calls[0] == (
        expander_module.DEFAULT_MODEL,
        {"local_files_only": True, "src_lang": "vi_VN"},
    )
    assert tokenizer.src_lang == "vi_VN"
    assert model.generate_kwargs["decoder_start_token_id"] == expected_decoder_id
    expected_num_beams = 3 if cuda else 1
    assert model.generate_kwargs["num_beams"] == expected_num_beams
    if cuda:
        assert model.generate_kwargs["early_stopping"] is True
    else:
        assert "early_stopping" not in model.generate_kwargs
    assert model.generate_kwargs["max_new_tokens"] == 64
    assert model.generate_kwargs["do_sample"] is False
    assert translated == "a speedboat finishes"


@pytest.mark.parametrize("cuda", [False, True])
def test_explicit_vinai_load_failure_falls_back_to_marian_without_decoder_token(
    monkeypatch, caplog, cuda
):
    fallback_tokenizer = FakeTokenizer(decoded="a runner")
    fallback_model = FakeModel()

    def tokenizer_factory(model_name, _kwargs):
        return FakeTokenizer() if model_name != expander_module.FALLBACK_MODEL else fallback_tokenizer

    def model_factory(model_name, _kwargs):
        if model_name != expander_module.FALLBACK_MODEL:
            raise OSError("primary unavailable")
        return fallback_model

    tokenizer_calls, model_calls = install_factories(
        monkeypatch, tokenizer_factory, model_factory, cuda=cuda
    )

    service = expander_module.LocalExpander(expander_module.DEFAULT_MODEL)
    assert service.translate("ng\u01b0\u1eddi ch\u1ea1y") == "a runner"

    assert service.model_name == expander_module.FALLBACK_MODEL
    assert [call[1]["local_files_only"] for call in model_calls] == [True, False, True]
    assert tokenizer_calls[-1] == (expander_module.FALLBACK_MODEL, {"local_files_only": True})
    assert "decoder_start_token_id" not in fallback_model.generate_kwargs
    assert "falling back" in caplog.text


def test_cleaning_cano_post_replacement_and_variants_deduplication(monkeypatch):
    cleaned, has_cano = expander_module.clean_vi_prompt("C\u00f3 c\u1ea3nh ca-n\u00f4 v\u1ec1 \u0111\u00edch")
    assert cleaned == "thuy\u1ec1n cao t\u1ed1c v\u1ec1 \u0111\u00edch"
    assert has_cano is True

    tokenizer = FakeTokenizer(decoded="speedbolat at sea.")
    model = FakeModel()
    install_factories(monkeypatch, lambda _name, _kwargs: tokenizer, lambda _name, _kwargs: model)
    service = expander_module.LocalExpander(expander_module.FALLBACK_MODEL)
    assert service.translate("ca-n\u00f4 v\u1ec1 \u0111\u00edch") == "speedboat at sea"

    service.translate = lambda _text: "CANO V\u1ec0 \u0110\u00cdCH"
    monkeypatch.setattr(expander_module, "_EXPANDER_INSTANCE", service)
    assert expander_module.variants("cano v\u1ec1 \u0111\u00edch") == ["cano v\u1ec1 \u0111\u00edch"]


@pytest.mark.parametrize("setting", ["1", "true", "yes"])
def test_cpu_quantization_is_opt_in(monkeypatch, setting):
    tokenizer = FakeTokenizer(language_ids={"en_XX": 42})
    model = FakeModuleModel()
    quantize_calls = []
    install_factories(monkeypatch, lambda _name, _kwargs: tokenizer, lambda _name, _kwargs: model)
    monkeypatch.setenv("SPATIAL_KIS_QUANTIZE_TRANSLATOR", setting)

    def quantize_dynamic(candidate, module_types, *, dtype):
        quantize_calls.append((candidate, module_types, dtype))
        return candidate

    monkeypatch.setattr(expander_module.torch.ao.quantization, "quantize_dynamic", quantize_dynamic)

    service = expander_module.LocalExpander(expander_module.DEFAULT_MODEL)
    service.translate("v\u1ec1 \u0111\u00edch")

    assert quantize_calls == [(model, {torch.nn.Linear}, torch.qint8)]
    assert model.generation_config.max_length is None
    assert model.generate_kwargs["num_beams"] == 1
    assert "early_stopping" not in model.generate_kwargs


@pytest.mark.parametrize("setting", [None, "0"])
def test_cpu_quantization_is_disabled_by_default_or_zero(monkeypatch, setting):
    tokenizer = FakeTokenizer()
    model = FakeModuleModel()
    install_factories(monkeypatch, lambda _name, _kwargs: tokenizer, lambda _name, _kwargs: model)
    if setting is not None:
        monkeypatch.setenv("SPATIAL_KIS_QUANTIZE_TRANSLATOR", setting)
    monkeypatch.setattr(
        expander_module.torch.ao.quantization,
        "quantize_dynamic",
        lambda *_args, **_kwargs: pytest.fail("quantization should be disabled"),
    )

    service = expander_module.LocalExpander(expander_module.FALLBACK_MODEL)
    service.translate("ng\u01b0\u1eddi ch\u1ea1y")

    assert model.generation_config.max_length is None
    assert model.generate_kwargs["num_beams"] == 1
    assert "early_stopping" not in model.generate_kwargs
