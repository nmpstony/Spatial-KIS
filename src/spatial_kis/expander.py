"""Vietnamese query expansion with Google Translate engine and local MarianMT fallback."""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.parse
import urllib.request
from functools import lru_cache

# Common Vietnamese prefixes in KIS descriptions that hurt CLIP embeddings if translated literally
_FILLER_PREFIXES = re.compile(
    r"^(có\s+cảnh\s+quay\s+|có\s+cảnh\s+|cảnh\s+quay\s+|cảnh\s+|hình\s+ảnh\s+|đoạn\s+video\s+có\s+|trong\s+cảnh\s+có\s+|tiếp\s+theo\s+là\s+cảnh\s+|sau\s+đó\s+là\s+cảnh\s+|có\s+một\s+|có\s+)",
    re.IGNORECASE,
)

_VI_PRE_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(cano|ca\s*nô|ca-nô)\b", re.IGNORECASE), "thuyền cao tốc"),
    (re.compile(r"\bflycam\b", re.IGNORECASE), "quay từ trên cao"),
    (re.compile(r"\b(mc|biên\s+tập\s+viên|btv)\b", re.IGNORECASE), "người dẫn chương trình truyền hình"),
    (re.compile(r"\báo\s+dài\b", re.IGNORECASE), "áo dài"),
    (re.compile(r"\bkhẩu\s+trang\b", re.IGNORECASE), "khẩu trang"),
]

_EN_POST_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bspeedbolat\b", re.IGNORECASE), "speedboat"),
    (re.compile(r"\bcane\b", re.IGNORECASE), "speedboat"),
    (re.compile(r"^there\s+is\s+a\s+scene\s+of\s+", re.IGNORECASE), ""),
    (re.compile(r"^there'?s\s+a\s+", re.IGNORECASE), "a "),
    (re.compile(r"^there\s+are\s+", re.IGNORECASE), ""),
    (re.compile(r"\bscene\s+at\s+sea\b", re.IGNORECASE), "on the sea"),
]

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def clean_vi_prompt(text: str) -> tuple[str, bool]:
    """Normalize Vietnamese KIS prompt and detect if 'cano' was mentioned."""
    cleaned = _FILLER_PREFIXES.sub("", text.strip()).strip()
    if not cleaned:
        cleaned = text.strip()
    has_cano = bool(re.search(r"\b(cano|ca\s*nô|ca-nô)\b", cleaned, re.IGNORECASE))
    for pattern, repl in _VI_PRE_REPLACEMENTS:
        cleaned = pattern.sub(repl, cleaned)
    return cleaned, has_cano


def _google_translate_raw(text: str, timeout: float = 4.0) -> str:
    """Translate Vietnamese text to English via Google Translate endpoints."""
    encoded = urllib.parse.quote(text.strip())
    
    # 1. Primary: translate.googleapis.com
    url1 = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=vi&tl=en&dt=t&q={encoded}"
    req1 = urllib.request.Request(url1, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req1, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and isinstance(data, list) and isinstance(data[0], list):
                translated = "".join(part[0] for part in data[0] if part and part[0]).strip()
                if translated:
                    return translated
    except Exception:
        pass

    # 2. Secondary fallback: clients5.google.com
    url2 = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=vi&tl=en&q={encoded}"
    req2 = urllib.request.Request(url2, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req2, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                return data[0].strip()
    except Exception:
        pass

    return ""


class LocalExpander:
    def __init__(self, model_name: str | None = None, engine: str | None = None) -> None:
        self.engine = (engine or os.getenv("SPATIAL_KIS_TRANSLATION_ENGINE", "google")).lower().strip()
        self.model_name = model_name or os.getenv("SPATIAL_KIS_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-vi-en")
        self._lock = threading.Lock()
        self._marian_loaded = False
        self.tokenizer = None
        self.model = None
        self.device = None

        if self.engine in ("opus", "marian"):
            self._ensure_marian()

    def _ensure_marian(self) -> bool:
        if self._marian_loaded:
            return True
        with self._lock:
            if self._marian_loaded:
                return True
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name, local_files_only=True
                    ).to(self.device).eval()
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()
                self._marian_loaded = True
                return True
            except Exception as e:
                print(f"Warning: MarianMT load failed: {e}", flush=True)
                return False

    def _marian_translate(self, text: str) -> str:
        if not self._ensure_marian():
            return ""
        import torch

        with self._lock:
            tokens = self.tokenizer([text], return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(**tokens, max_new_tokens=48, num_beams=1, do_sample=False)
            return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    @lru_cache(maxsize=4096)
    def translate(self, text: str) -> str:
        cleaned, has_cano = clean_vi_prompt(text)
        translated = ""

        # Use Google Translate when engine is google (default)
        if self.engine == "google":
            try:
                translated = _google_translate_raw(cleaned)
            except Exception as e:
                print(f"Warning: Google Translate failed: {e}", flush=True)

            # Offline fallback to MarianMT if Google Translate was unreachable
            if not translated:
                try:
                    translated = self._marian_translate(cleaned)
                except Exception:
                    pass
        else:
            translated = self._marian_translate(cleaned)

        if not translated:
            return ""

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
