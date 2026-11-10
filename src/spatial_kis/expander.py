"""Offline-only Vietnamese query expansion with visual prompt normalization."""
from __future__ import annotations

import os
import re
from functools import lru_cache
import torch

# Common Vietnamese prefixes in KIS descriptions that hurt CLIP embeddings if translated literally
_FILLER_PREFIXES = re.compile(
    r"^(có\s+cảnh\s+quay\s+|có\s+cảnh\s+|cảnh\s+quay\s+|cảnh\s+|hình\s+ảnh\s+|đoạn\s+video\s+có\s+|trong\s+cảnh\s+có\s+|tiếp\s+theo\s+là\s+cảnh\s+|sau\s+đó\s+là\s+cảnh\s+|có\s+một\s+|có\s+)",
    re.IGNORECASE,
)

# Vietnamese terms that MarianMT (opus-mt-vi-en) commonly mistranslates
_VI_PRE_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(cano|ca\s*nô|ca-nô)\b", re.IGNORECASE), "speedboat"),
    (re.compile(r"\bflycam\b", re.IGNORECASE), "aerial view from above"),
    (re.compile(r"\b(mc|biên\s+tập\s+viên|btv)\b", re.IGNORECASE), "người dẫn chương trình truyền hình"),
    (re.compile(r"\báo\s+dài\b", re.IGNORECASE), "trang phục áo dài truyền thống"),
    (re.compile(r"\bkhẩu\s+trang\b", re.IGNORECASE), "khẩu trang y tế che mặt"),
]

_EN_POST_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcane\b", re.IGNORECASE), "speedboat motorboat"),
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
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name or os.getenv("SPATIAL_KIS_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-vi-en")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()

    def translate(self, text: str) -> str:
        cleaned, has_cano = clean_vi_prompt(text)
        tokens = self.tokenizer([cleaned], return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**tokens, max_new_tokens=64)
        translated = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        translated = re.sub(r"[.!?]+$", "", translated).strip()
        for pattern, repl in _EN_POST_REPLACEMENTS:
            translated = pattern.sub(repl, translated).strip()
        if has_cano and "boat" not in translated.lower() and "speedboat" not in translated.lower():
            translated = f"speedboat motorboat {translated}".strip()
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
        except Exception:
            pass
        return list(dict.fromkeys(outputs))


@lru_cache
def expander() -> LocalExpander | None:
    try:
        return LocalExpander()
    except Exception as e:
        print(f"Warning: LocalExpander init failed: {e}")
        return None


def variants(text: str) -> list[str]:
    service = expander()
    return service.expand(text) if service else [text]
