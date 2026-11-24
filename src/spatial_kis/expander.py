"""Offline-only Vietnamese query expansion with thread-safe initialization and visual prompt normalization."""
from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "vinai/vinai-translate-vi2en-v2"
FALLBACK_MODEL = "Helsinki-NLP/opus-mt-vi-en"
logger = logging.getLogger(__name__)


# Common Vietnamese prefixes in KIS descriptions that hurt CLIP embeddings if translated literally
_FILLER_PREFIXES = re.compile(
    r"^(có\s+cảnh\s+quay\s+|có\s+cảnh\s+|cảnh\s+quay\s+|cảnh\s+|hình\s+ảnh\s+|đoạn\s+video\s+có\s+|trong\s+cảnh\s+có\s+|tiếp\s+theo\s+là\s+cảnh\s+|sau\s+đó\s+là\s+cảnh\s+|có\s+một\s+|có\s+)",
    re.IGNORECASE,
)

# Vietnamese terms that MarianMT (opus-mt-vi-en) commonly mistranslates (must remain pure Vietnamese here)
_VI_PRE_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(cano|ca\s*nô|ca-nô)\b", re.IGNORECASE), "thuyền cao tốc"),
    (re.compile(r"\bflycam\b", re.IGNORECASE), "quay từ trên cao"),
    (re.compile(r"\b(mc|biên\s+tập\s+viên|btv)\b", re.IGNORECASE), "người dẫn chương trình truyền hình"),
    (re.compile(r"\báo\s+dài\b", re.IGNORECASE), "trang phục áo dài truyền thống"),
    (re.compile(r"\bkhẩu\s+trang\b", re.IGNORECASE), "khẩu trang che mặt"),
]

_EN_POST_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bspeedbolat\b", re.IGNORECASE), "speedboat"),
    (re.compile(r"\bcane\b", re.IGNORECASE), "speedboat"),
    (re.compile(r"^there\s+is\s+a\s+scene\s+of\s+", re.IGNORECASE), ""),
    (re.compile(r"^there'?s\s+a\s+", re.IGNORECASE), "a "),
    (re.compile(r"^there\s+are\s+", re.IGNORECASE), ""),
    (re.compile(r"\bscene\s+at\s+sea\b", re.IGNORECASE), "on the sea"),
]


def clean_vi_prompt(text: str) -> tuple[str, bool]:
    """Normalize Vietnamese KIS prompt and detect if 'cano' was mentioned."""
    cleaned = _FILLER_PREFIXES.sub("", text.strip()).strip()
    if not cleaned:
        cleaned = text.strip()
    has_cano = bool(re.search(r"\b(cano|ca\s*nô|ca-nô)\b", cleaned, re.IGNORECASE))
    for pattern, repl in _VI_PRE_REPLACEMENTS:
        cleaned = pattern.sub(repl, cleaned)
    return cleaned, has_cano


class LocalExpander:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("SPATIAL_KIS_TRANSLATION_MODEL", DEFAULT_MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        try:
            self.tokenizer, self.model = self._load(self.model_name)
        except Exception as primary_error:
            if self.model_name == FALLBACK_MODEL:
                raise
            logger.warning(
                "Could not load translation model %s; falling back to %s: %s",
                self.model_name,
                FALLBACK_MODEL,
                primary_error,
            )
            self.tokenizer, self.model = self._load(FALLBACK_MODEL)
            self.model_name = FALLBACK_MODEL

        self.src_code, self.tgt_code = self._language_codes(self.model_name, self.tokenizer)

    @staticmethod
    def _language_codes(model_name: str, tokenizer=None) -> tuple[str | None, str | None]:
        lowered = model_name.lower()
        if "nllb" in lowered:
            return "vie_Latn", "eng_Latn"
        if (
            "vinai-translate" in lowered
            or "mbart" in lowered
            or isinstance(getattr(tokenizer, "lang_code_to_id", None), dict)
        ):
            return "vi_VN", "en_XX"
        return None, None

    def _load(self, model_name: str):
        src_code, _ = self._language_codes(model_name)
        tokenizer_kwargs = {"src_lang": src_code} if src_code else {}
        last_error: Exception | None = None
        for local_files_only in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                    **tokenizer_kwargs,
                )
                if src_code and hasattr(tokenizer, "src_lang"):
                    tokenizer.src_lang = src_code
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                ).to(self.device).eval()
                return tokenizer, model
            except Exception as error:
                last_error = error
        assert last_error is not None
        raise last_error

    def _decoder_start_token_id(self) -> int | None:
        if self.tgt_code is None:
            return None
        language_ids = getattr(self.tokenizer, "lang_code_to_id", None)
        if isinstance(language_ids, dict) and self.tgt_code in language_ids:
            return language_ids[self.tgt_code]
        get_vocab = getattr(self.tokenizer, "get_vocab", None)
        vocab = get_vocab() if callable(get_vocab) else {}
        if self.tgt_code not in vocab:
            return None
        token_id = vocab.get(self.tgt_code)
        if token_id is None:
            convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            token_id = convert(self.tgt_code) if callable(convert) else None
        return token_id

    @lru_cache(maxsize=1024)
    def translate(self, text: str) -> str:
        cleaned, has_cano = clean_vi_prompt(text)
        with self._lock:
            tokens = self.tokenizer([cleaned], return_tensors="pt", truncation=True).to(self.device)
            generate_kwargs = {
                "max_new_tokens": 64,
                "num_beams": 3,
                "early_stopping": True,
                "do_sample": False,
            }
            decoder_start_token_id = self._decoder_start_token_id()
            if decoder_start_token_id is not None:
                generate_kwargs["decoder_start_token_id"] = decoder_start_token_id
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    generated = self.model.generate(**tokens, **generate_kwargs)
            translated = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        translated = re.sub(r"[.!?]+$", "", translated).strip()
        for pattern, repl in _EN_POST_REPLACEMENTS:
            translated = pattern.sub(repl, translated).strip()
        if has_cano and "boat" not in translated.lower() and "speedboat" not in translated.lower():
            translated = f"speedboat {translated}".strip()
        return translated

    def expand(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        outputs = [text]
        try:
            translated = self.translate(text)
            if translated and translated.lower() != text.lower():
                outputs.append(translated)
        except Exception as e:
            print(f"Warning: LocalExpander.translate failed: {e}", flush=True)
        return list(dict.fromkeys(outputs))


_EXPANDER_INSTANCE: LocalExpander | None = None
_EXPANDER_LOCK = threading.Lock()


def expander() -> LocalExpander | None:
    global _EXPANDER_INSTANCE
    if _EXPANDER_INSTANCE is not None:
        return _EXPANDER_INSTANCE
    with _EXPANDER_LOCK:
        if _EXPANDER_INSTANCE is None:
            try:
                _EXPANDER_INSTANCE = LocalExpander()
            except Exception as e:
                print(f"Warning: LocalExpander init failed: {e}", flush=True)
                return None
    return _EXPANDER_INSTANCE


def variants(text: str) -> list[str]:
    service = expander()
    return service.expand(text) if service else [text]
