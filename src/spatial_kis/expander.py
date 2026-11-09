"""Offline-only Vietnamese query expansion."""
from __future__ import annotations

import os
import re
from functools import lru_cache
import torch


class LocalExpander:
    def __init__(self, model_name: str | None = None) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name or os.getenv("SPATIAL_KIS_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-vi-en")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()

    def translate(self, text: str) -> str:
        tokens = self.tokenizer([text], return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**tokens, max_new_tokens=64)
        translated = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        translated = re.sub(r"[.!?]+$", "", translated).strip()
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
