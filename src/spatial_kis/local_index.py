from __future__ import annotations

import json
import fnmatch
import logging
import pickle
import re
import threading
import unicodedata
from collections import defaultdict
from functools import update_wrapper
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .artifact import iter_manifest
from .ranking import Hit, diversify_by_shot, normalized_ip
from .regions import region_ids


def clean_video_name(video_id: str) -> str:
    """Return DRES-compatible mediaItemName (video filename without .mp4 extension)."""
    name = Path(video_id).name
    return name[:-4] if name.lower().endswith(".mp4") else name


class _BoundPerInstanceCache:
    def __init__(self, descriptor: "_PerInstanceCache", instance: object) -> None:
        self._descriptor = descriptor
        self._instance = instance

    def __call__(self, region_id: int = 0) -> torch.Tensor:
        return self._descriptor._call(self._instance, region_id)

    def cache_clear(self) -> None:
        self._descriptor._clear(self._instance)


class _PerInstanceCache:
    """One-entry, per-instance cache exposing a bound ``cache_clear`` method."""

    def __init__(self, fn) -> None:
        self._fn = fn
        update_wrapper(self, fn)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return _BoundPerInstanceCache(self, instance)

    def _call(self, instance, region_id: int) -> torch.Tensor:
        with instance._fp32_lock:
            cached = instance._fp32_cache.get(region_id)
            if cached is not None:
                return cached
            value = self._fn(instance, region_id)
            instance._fp32_cache.clear()
            instance._fp32_cache[region_id] = value
            return value

    @staticmethod
    def _clear(instance) -> None:
        with instance._fp32_lock:
            instance._fp32_cache.clear()


class LocalMemmapIndex:
    """Zero-Docker memory-mapped exact vector search engine for Spatial-KIS."""

    def __init__(self, artifact_root: Path) -> None:
        self._fp32_lock = threading.Lock()
        self._fp32_cache: dict[int, torch.Tensor] = {}
        self.artifact_root = artifact_root
        self.index_dir = artifact_root / "memmap_index"
        self.meta_path = self.index_dir / "meta.json"
        self.records_path = self.index_dir / "records.pkl"
        if not self.meta_path.exists() or not self.records_path.exists():
            self.build()
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.count: int = int(meta["count"])
        self.dim: int = int(meta["dim"])
        with self.records_path.open("rb") as handle:
            data = pickle.load(handle)
        self.records: list[dict] = data["records"]
        self.by_keyframe: dict[str, int] = data["by_keyframe"]
        self.by_video: dict[str, list[int]] = data["by_video"]
        self._mmaps: dict[int, np.ndarray] = {}
        for region_id in range(10):
            path = self.index_dir / f"region_{region_id}.fp16.bin"
            self._mmaps[region_id] = np.memmap(path, dtype=np.float16, mode="r", shape=(self.count, self.dim))

    def build(self) -> None:
        print(f"[LocalMemmapIndex] Building memory-mapped index from {self.artifact_root}...", flush=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        manifest_records = list(iter_manifest(self.artifact_root))
        if not manifest_records:
            raise RuntimeError(f"No keyframes found in {self.artifact_root / 'manifest.jsonl'}")

        first_shard = np.load(self.artifact_root / manifest_records[0].embedding_shard)["vectors"]
        dim = int(first_shard.shape[-1])
        count = len(manifest_records)

        mmaps: list[np.memmap] = []
        for region_id in range(10):
            path = self.index_dir / f"region_{region_id}.fp16.bin"
            mmaps.append(np.memmap(path, dtype=np.float16, mode="w+", shape=(count, dim)))

        by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
        compact_records: list[dict] = []
        by_keyframe: dict[str, int] = {}
        by_video_raw: dict[str, list[int]] = defaultdict(list)

        for idx, rec in enumerate(manifest_records):
            by_shard[rec.embedding_shard].append((idx, rec.embedding_offset))
            video_stem = clean_video_name(rec.video_id)
            item = {
                "keyframe_id": rec.keyframe_id,
                "video_id": rec.video_id,
                "video_stem": video_stem,
                "shot_id": int(rec.shot_id),
                "frame_id": int(rec.frame_id),
                "fps": float(rec.fps),
                "timestamp": float(rec.timestamp),
                "timestamp_ms": int(round(float(rec.timestamp) * 1000)),
                "image_path": rec.image_path,
            }
            compact_records.append(item)
            by_keyframe[rec.keyframe_id] = idx
            by_video_raw[video_stem].append(idx)
            if rec.video_id != video_stem:
                by_video_raw[rec.video_id].append(idx)

        total_shards = len(by_shard)
        for shard_idx, (shard_name, entries) in enumerate(by_shard.items(), 1):
            vectors = np.load(self.artifact_root / shard_name)["vectors"].astype(np.float16)
            indices = [idx for idx, _ in entries]
            offsets = [off for _, off in entries]
            shard_slice = vectors[offsets]  # shape: [K, 10, dim]
            for region_id in range(10):
                mmaps[region_id][indices] = shard_slice[:, region_id, :]
            if shard_idx % 100 == 0 or shard_idx == total_shards:
                print(f"  [LocalMemmapIndex] Processed {shard_idx}/{total_shards} shards...", flush=True)

        for mm in mmaps:
            mm.flush()

        by_video: dict[str, list[int]] = {}
        for key, idxs in by_video_raw.items():
            by_video[key] = sorted(idxs, key=lambda i: compact_records[i]["timestamp"])

        with self.records_path.open("wb") as handle:
            pickle.dump(
                {"records": compact_records, "by_keyframe": by_keyframe, "by_video": by_video},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self.meta_path.write_text(
            json.dumps({"count": count, "dim": dim, "version": 1}, indent=2),
            encoding="utf-8",
        )
        print(f"[LocalMemmapIndex] Built {count:,} keyframes x 10 regions ({dim}D).", flush=True)

    @_PerInstanceCache
    def _region_tensor_fp32(self, region_id: int = 0) -> torch.Tensor:
        """Keep region 0 (full frame) permanently in float32 RAM (~1.29 GB) for sub-100ms BLAS matmul."""
        arr = np.array(self._mmaps[region_id], dtype=np.float32, copy=True)
        return torch.from_numpy(arr)

    def _matmul_region(self, region_id: int, q_tensor: torch.Tensor) -> torch.Tensor:
        """Multiply region [N, dim] by q_tensor [dim, M] without evicting OS page cache."""
        if region_id == 0:
            return torch.matmul(self._region_tensor_fp32(0), q_tensor)
        mmap_arr = self._mmaps[region_id]
        n = self.count
        m = q_tensor.shape[1]
        out = torch.empty((n, m), dtype=torch.float32)
        step = 131072
        for start in range(0, n, step):
            end = min(start + step, n)
            chunk = torch.from_numpy(np.array(mmap_arr[start:end], dtype=np.float32, copy=True))
            torch.matmul(chunk, q_tensor, out=out[start:end])
        return out

    def score_region(self, region_id: int, query_vectors: list[list[float]]) -> np.ndarray:
        """Return max inner-product score per keyframe across query_vectors for a single region."""
        q = torch.tensor(query_vectors, dtype=torch.float32).T  # [dim, V]
        with torch.no_grad():
            scores = self._matmul_region(region_id, q)  # [N, V]
            if scores.shape[1] == 1:
                return scores[:, 0].numpy()
            return scores.max(dim=1).values.numpy()

    def score_condition(self, regions: Iterable[int], query_vectors: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        """Return (best_scores [N], best_region_ids [N]) across all selected regions in a condition."""
        region_list = list(regions)
        best_scores: np.ndarray | None = None
        best_regions: np.ndarray | None = None
        for reg in region_list:
            reg_scores = self.score_region(reg, query_vectors)
            if best_scores is None:
                best_scores = reg_scores.copy()
                best_regions = np.full(self.count, reg, dtype=np.int8)
            else:
                mask = reg_scores > best_scores
                best_scores[mask] = reg_scores[mask]
                best_regions[mask] = reg
        assert best_scores is not None and best_regions is not None
        return best_scores, best_regions

    def resolve_allowed_videos(self, raw_filters: list[str] | None) -> set[str] | None:
        if not raw_filters:
            return None
        tokens: list[str] = []
        for item in raw_filters:
            for part in item.replace(";", ",").replace(" ", ",").split(","):
                cleaned = clean_video_name(part.strip()).upper()
                if not cleaned:
                    continue
                # Support range syntax like L21-L24
                if "-" in cleaned and len(cleaned.split("-")) == 2:
                    p1, p2 = cleaned.split("-")
                    if len(p1) == 3 and len(p2) == 3 and p1[0] == p2[0] and p1[1:].isdigit() and p2[1:].isdigit():
                        prefix = p1[0]
                        for b_num in range(int(p1[1:]), int(p2[1:]) + 1):
                            tokens.append(f"{prefix}{b_num:02d}")
                        continue
                tokens.append(cleaned)
        if not tokens:
            return None
        token_set = set(tokens)
        full_stem_re = re.compile(r"^L\d{2}_V\d{3}$")
        prefix_tokens = [tok for tok in token_set if not full_stem_re.fullmatch(tok)]
        allowed: set[str] = set()
        for stem in self.by_video:
            clean_st = clean_video_name(stem)
            stem_up = clean_st.upper()
            if stem_up in token_set or any(stem_up.startswith(tok) for tok in prefix_tokens):
                allowed.add(clean_st)
        return allowed

    def search_conditions(
        self,
        evaluated_conditions: list[tuple[tuple[int, ...], list[list[float]], float, int]],
        top_k: int,
        result_limit: int,
        diversify: bool,
        max_temporal_gap: float = 45.0,
        allowed_videos: set[str] | None = None,
    ) -> dict:
        """
        Full-corpus multi-condition & temporal search.
        Each item in evaluated_conditions is: (region_ids_tuple, query_vectors, weight, stage).
        stage=1: primary scene; stage=2: subsequent scene within max_temporal_gap seconds in the same video.
        """
        # Pre-batch all matrix multiplications by region_id so each 1.3GB region tensor is scanned only ONCE per search
        region_queries: dict[int, list[list[float]]] = defaultdict(list)
        cond_region_slices: list[dict[int, tuple[int, int]]] = []
        for regions_tuple, q_vecs, _, _ in evaluated_conditions:
            slices_for_cond: dict[int, tuple[int, int]] = {}
            for reg in regions_tuple:
                start_col = len(region_queries[reg])
                region_queries[reg].extend(q_vecs)
                slices_for_cond[reg] = (start_col, len(region_queries[reg]))
            cond_region_slices.append(slices_for_cond)

        region_score_matrices: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for reg, q_list in region_queries.items():
                q_tensor = torch.tensor(q_list, dtype=torch.float32).T  # [dim, M]
                region_score_matrices[reg] = self._matmul_region(reg, q_tensor)  # [N, M]

        def eval_single_cond(cond_idx: int) -> tuple[np.ndarray, np.ndarray]:
            slices_for_cond = cond_region_slices[cond_idx]
            best_scores: np.ndarray | None = None
            best_regions: np.ndarray | None = None
            for reg, (c0, c1) in slices_for_cond.items():
                sub = region_score_matrices[reg][:, c0:c1]
                reg_scores = sub[:, 0].numpy() if (c1 - c0) == 1 else sub.max(dim=1).values.numpy()
                if best_scores is None:
                    best_scores = reg_scores.copy()
                    best_regions = np.full(self.count, reg, dtype=np.int8)
                else:
                    mask = reg_scores > best_scores
                    best_scores[mask] = reg_scores[mask]
                    best_regions[mask] = reg
            assert best_scores is not None and best_regions is not None
            return best_scores, best_regions

        unique_stages = sorted({max(1, int(item[3])) for item in evaluated_conditions})
        stage_groups: list[tuple[int, list[int]]] = [
            (st, [i for i, item in enumerate(evaluated_conditions) if max(1, int(item[3])) == st])
            for st in unique_stages
        ]

        def normalize_distribution(scores: np.ndarray) -> np.ndarray:
            if self.count < 16:
                return np.clip((scores + 1.0) * 0.5, 1e-6, 1.0)
            mean_val = float(np.mean(scores))
            span = float(np.max(scores)) - mean_val
            if span <= 1e-6:
                return np.clip((scores + 1.0) * 0.5, 1e-6, 1.0)
            return np.clip((scores - mean_val) / span, 1e-6, 1.0)

        def combine_stage(cond_idxs: list[int]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
            raw_scores_list: list[np.ndarray] = []
            norm_scores_list: list[np.ndarray] = []
            weights: list[float] = []
            primary_regions: np.ndarray | None = None
            for c_idx in cond_idxs:
                weight = evaluated_conditions[c_idx][2]
                scores, regs = eval_single_cond(c_idx)
                raw_scores_list.append(scores)
                norm_scores_list.append(normalize_distribution(scores))
                weights.append(float(weight))
                if primary_regions is None:
                    primary_regions = regs
            assert primary_regions is not None
            if len(cond_idxs) == 1:
                combined = norm_scores_list[0]
            else:
                w_sum = sum(weights)
                denom = np.zeros(self.count, dtype=np.float32)
                for norm_s, w in zip(norm_scores_list, weights):
                    denom += w / norm_s
                combined = w_sum / denom
            return combined, primary_regions, raw_scores_list

        stage_evals = [combine_stage(c_idxs) for _, c_idxs in stage_groups]
        stage_weights = [
            sum(float(evaluated_conditions[i][2]) for i in c_idxs) for _, c_idxs in stage_groups
        ]
        s1_combined, s1_regions, s1_raw = stage_evals[0]

        matched_chains: dict[int, list[tuple[int, int]]] = {}
        num_stages = len(stage_groups)
        if num_stages >= 2:
            final_scores = np.zeros(self.count, dtype=np.float32)
            if allowed_videos is not None:
                cand_videos = set(allowed_videos)
            else:
                cand_k = min(12000, self.count)
                stage_video_sets: list[set[str]] = []
                for st_comb, _, _ in stage_evals:
                    top_st = np.argpartition(st_comb, -cand_k)[-cand_k:]
                    stage_video_sets.append({self.records[int(i)]["video_stem"] for i in top_st})
                cand_videos = set.intersection(*stage_video_sets) if stage_video_sets else set()
                if not cand_videos and stage_video_sets:
                    cand_videos = set.union(*stage_video_sets)

            w_total_all = sum(stage_weights)
            inf = float("inf")
            if allowed_videos is not None and len(allowed_videos) <= 40:
                min_stage_norm = 0.12
            else:
                min_stage_norm = 0.32 if self.count >= 1000 else 0.15
            for stem in cand_videos:
                v_idxs = self.by_video.get(stem, [])
                n_v = len(v_idxs)
                if n_v < num_stages:
                    continue
                times = [self.records[i]["timestamp"] for i in v_idxs]
                st_vals = [[float(stage_evals[s][0][i]) for i in v_idxs] for s in range(num_stages)]

                # Backward DP over stages s = num_stages-1 down to 0 to minimize harmonic denominator sum(W_s / score_s)
                dp_denom = [[inf] * n_v for _ in range(num_stages)]
                next_hop = [[-1] * n_v for _ in range(num_stages)]

                w_last = stage_weights[-1]
                for a in range(n_v):
                    val = st_vals[-1][a]
                    if val >= min_stage_norm:
                        dp_denom[-1][a] = w_last / val

                for s in range(num_stages - 2, -1, -1):
                    w_s = stage_weights[s]
                    vals_s = st_vals[s]
                    next_denom = dp_denom[s + 1]
                    for a in range(n_v):
                        val_a = vals_s[a]
                        if val_a < min_stage_norm:
                            continue
                        t_a = times[a]
                        best_b = -1
                        best_d = inf
                        for b in range(a + 1, n_v):
                            dt = times[b] - t_a
                            if dt > max_temporal_gap:
                                break
                            if dt >= 0.5 and next_denom[b] < best_d:
                                best_d = next_denom[b]
                                best_b = b
                        if best_b >= 0 and best_d < inf:
                            dp_denom[s][a] = (w_s / val_a) + best_d
                            next_hop[s][a] = best_b

                for a in range(n_v):
                    d_total = dp_denom[0][a]
                    if d_total < inf:
                        idx_a = v_idxs[a]
                        final_scores[idx_a] = w_total_all / d_total
                        chain: list[tuple[int, int]] = []
                        curr = a
                        for s in range(num_stages - 1):
                            nxt = next_hop[s][curr]
                            if nxt < 0:
                                break
                            chain.append((stage_groups[s + 1][0], v_idxs[nxt]))
                            curr = nxt
                        if chain:
                            matched_chains[idx_a] = chain
        else:
            if allowed_videos is not None:
                final_scores = np.zeros(self.count, dtype=np.float32)
                for stem in allowed_videos:
                    for idx_a in self.by_video.get(stem, []):
                        final_scores[idx_a] = s1_combined[idx_a]
            else:
                final_scores = s1_combined

        k = min(max(top_k, result_limit * 4), self.count)
        top_indices = np.argpartition(final_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

        ranked: list[dict] = []
        for idx in top_indices:
            idx_int = int(idx)
            sc = float(final_scores[idx_int])
            if sc <= 0.0:
                continue
            rec = self.records[idx_int]
            if allowed_videos is not None and rec["video_stem"] not in allowed_videos:
                continue
            entry = {
                **rec,
                "score": sc,
                "region_id": int(s1_regions[idx_int]),
                "condition_scores": [float(arr[idx_int]) for arr in s1_raw],
            }
            if idx_int in matched_chains:
                next_list = []
                for st_num, m_idx in matched_chains[idx_int]:
                    next_rec = self.records[m_idx]
                    next_list.append({
                        "stage": st_num,
                        "keyframe_id": next_rec["keyframe_id"],
                        "frame_id": next_rec["frame_id"],
                        "timestamp": next_rec["timestamp"],
                        "timestamp_ms": next_rec["timestamp_ms"],
                        "image_path": next_rec["image_path"],
                    })
                entry["next_matches"] = next_list
                entry["next_match"] = next_list[0]
            ranked.append(entry)

        results = diversify_by_shot(ranked, result_limit) if diversify else ranked[:result_limit]
        return {"kind": "frames", "results": results, "raw_count": len(ranked)}

    def video_neighbors(self, video_id: str, timestamp: float, window_seconds: float = 30.0, limit: int = 24) -> list[dict]:
        """Return neighboring keyframes from the same video around timestamp."""
        stem = clean_video_name(video_id)
        idxs = self.by_video.get(stem) or self.by_video.get(video_id) or []
        if not idxs:
            return []
        nearby = [
            self.records[i]
            for i in idxs
            if abs(self.records[i]["timestamp"] - timestamp) <= window_seconds
        ]
        if len(nearby) > limit:
            nearby.sort(key=lambda r: abs(r["timestamp"] - timestamp))
            nearby = sorted(nearby[:limit], key=lambda r: r["timestamp"])
        return nearby


logger = logging.getLogger(__name__)

_OCR_CONFUSIONS = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
})
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_NON_ALNUM_KEEP_PIPE_RE = re.compile(r"[^a-z0-9|]+")

# Conservative EasyOCR visual confusion pairs (weight 0.30 instead of 1.00)
_CONFUSION_PAIRS: frozenset[tuple[str, str]] = frozenset(
    pair
    for a, b in (
        ("o", "q"), ("o", "d"), ("q", "d"),
        ("i", "l"), ("i", "j"), ("l", "j"), ("i", "t"), ("j", "t"),
        ("u", "v"), ("b", "v"), ("n", "m"), ("c", "e"),
        ("a", "o"), ("h", "k"),
    )
    for pair in ((a, b), (b, a))
)

# Connected-component bitmask & canonical signature for O(1) unit-edit prefiltering
_GROUP_BITS: dict[str, int] = {}
_GROUP_CHAR_TRANS: dict[int, str] = {}
_SIG_ALPHABET = "0123456789abcdef"
for _bit_idx, _grp in enumerate((
    "aoqd", "iljt", "uvb", "nm", "ce", "hk",
    "f", "g", "p", "r", "s", "w", "x", "y", "z",
)):
    _sig_ch = _SIG_ALPHABET[_bit_idx]
    for _ch in _grp:
        _GROUP_BITS[_ch] = 1 << _bit_idx
        _GROUP_CHAR_TRANS[ord(_ch)] = _sig_ch
for _ch in "0123456789":
    _GROUP_BITS[_ch] = 1 << 15
    _GROUP_CHAR_TRANS[ord(_ch)] = _ch


def _conf_mask(s: str) -> int:
    m = 0
    gb = _GROUP_BITS
    for ch in s:
        m |= gb.get(ch, 0)
    return m


def _conf_sig(s: str) -> str:
    """Map characters in each visual-confusion group to a single canonical symbol."""
    return s.translate(_GROUP_CHAR_TRANS)


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").lower()


def ocr_canonical(text: str) -> str:
    """Normalize Vietnamese accents and common OCR digit/symbol confusions into clean space-separated tokens."""
    norm = strip_accents(text).translate(_OCR_CONFUSIONS)
    return _NON_ALNUM_RE.sub(" ", norm).strip()


def _ocr_segments_and_text(text_norm: str) -> tuple[str, tuple[tuple[str, ...], ...]]:
    translated = text_norm.translate(_OCR_CONFUSIONS)
    clean_pipe = _NON_ALNUM_KEEP_PIPE_RE.sub(" ", translated)
    segments = tuple(
        toks
        for seg in clean_pipe.split("|")
        if (toks := tuple(seg.split()))
    )
    text_ocr = " ".join(" ".join(seg) for seg in segments)
    return text_ocr, segments


_ASR_TRANSLITERATIONS = str.maketrans(
    {"œ": "oe", "æ": "ae", "Œ": "OE", "Æ": "AE"}
)


def _asr_aligned_segments(
    text: str,
) -> tuple[str, tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Tokenize ASR text into permanently aligned normalized/raw token pairs."""
    text = unicodedata.normalize("NFC", text)
    raw_pipe = "".join(
        char.lower() if (char.isalnum() or char == "|") else " "
        for char in text
    )
    norm_segments: list[tuple[str, ...]] = []
    raw_segments: list[tuple[str, ...]] = []
    for raw_segment in raw_pipe.split("|"):
        norm_tokens: list[str] = []
        kept_raw_tokens: list[str] = []
        for raw_token in raw_segment.split():
            normalized = strip_accents(raw_token.translate(_ASR_TRANSLITERATIONS))
            normalized = _NON_ALNUM_RE.sub("", normalized)
            if normalized:
                norm_tokens.append(normalized)
                kept_raw_tokens.append(raw_token)
        if norm_tokens:
            norm_segment = tuple(norm_tokens)
            kept_raw_segment = tuple(kept_raw_tokens)
            assert len(norm_segment) == len(kept_raw_segment)
            norm_segments.append(norm_segment)
            raw_segments.append(kept_raw_segment)
    segments = tuple(norm_segments)
    aligned_raw_segments = tuple(raw_segments)
    assert len(segments) == len(aligned_raw_segments)
    return " ".join(" ".join(seg) for seg in segments), segments, aligned_raw_segments


def _has_diacritics(token: str) -> bool:
    return token.lower() != strip_accents(token)


def _asr_accent_conflicts(
    query_raw_tokens: tuple[str, ...],
    query_has_diacritics: tuple[bool, ...],
    candidate_tokens: tuple[str, ...],
    candidate_raw_tokens: tuple[str, ...],
) -> int:
    """Count differing explicit accents without normalizing inside the hot loop."""
    return sum(
        1
        for query_raw, query_has, candidate, candidate_raw in zip(
            query_raw_tokens,
            query_has_diacritics,
            candidate_tokens,
            candidate_raw_tokens,
        )
        if query_has
        and candidate_raw != candidate
        and query_raw != candidate_raw
    )


def _asr_conflicts_allowed(query_tokens: tuple[str, ...]) -> int:
    return 1 if len(query_tokens) >= 4 else 0


def _asr_contiguous_accent_match(
    q_words: tuple[str, ...],
    q_raw_words: tuple[str, ...],
    q_has_diacritics: tuple[bool, ...],
    row_segments: tuple[tuple[str, ...], ...],
    raw_segments: tuple[tuple[str, ...], ...],
) -> bool:
    """Find a normalized contiguous match with an acceptable accent pattern."""
    if not q_words:
        return False
    allowed = _asr_conflicts_allowed(q_raw_words)
    for segment, raw_segment in zip(row_segments, raw_segments):
        width = len(q_words)
        for start in range(len(segment) - width + 1):
            normalized_match = (
                q_words[0] in segment[start]
                if width == 1
                else segment[start : start + width] == q_words
            )
            if not normalized_match:
                continue
            candidate_window = segment[start : start + width]
            raw_window = raw_segment[start : start + width]
            if _asr_accent_conflicts(
                q_raw_words, q_has_diacritics, candidate_window, raw_window
            ) <= allowed:
                return True
    return False


def _unit_edit_cost(a: str, b: str, max_edits: int = 1) -> float:
    """Unweighted Levenshtein distance with a small bounded-distance fast path."""
    if abs(len(a) - len(b)) > max_edits:
        return float(max_edits + 1)
    if a == b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for j, char_b in enumerate(b, 1):
        current = [j]
        row_min = j
        for i, char_a in enumerate(a, 1):
            value = min(
                current[-1] + 1,
                previous[i] + 1,
                previous[i - 1] + (char_a != char_b),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_edits:
            return float(max_edits + 1)
        previous = current
    return float(previous[-1])


def _asr_tokens_in_order(
    q_words: tuple[str, ...],
    row_segments: tuple[tuple[str, ...], ...],
    max_gap: int = 2,
    q_raw_words: tuple[str, ...] | None = None,
    q_has_diacritics: tuple[bool, ...] | None = None,
    raw_segments: tuple[tuple[str, ...], ...] | None = None,
) -> bool:
    """Match tokens in one segment with at most ``max_gap`` filler tokens per step."""
    if len(q_words) < 2:
        return False
    allowed_conflicts = _asr_conflicts_allowed(q_raw_words or ())
    for segment_index, segment in enumerate(row_segments):
        if len(segment) < len(q_words):
            continue
        for first_pos, token in enumerate(segment):
            if token != q_words[0]:
                continue
            paths = [(first_pos,)]
            for query_token in q_words[1:]:
                next_paths: list[tuple[int, ...]] = []
                for path in paths:
                    previous = path[-1]
                    stop = min(len(segment), previous + max_gap + 2)
                    next_paths.extend(
                        path + (pos,)
                        for pos in range(previous + 1, stop)
                        if segment[pos] == query_token
                    )
                paths = next_paths
                if not paths:
                    break
            if not paths:
                continue
            for positions in paths:
                if q_raw_words is not None and raw_segments is not None:
                    raw_segment = raw_segments[segment_index]
                    candidate = tuple(segment[pos] for pos in positions)
                    candidate_raw = tuple(raw_segment[pos] for pos in positions)
                    if _asr_accent_conflicts(
                        q_raw_words,
                        q_has_diacritics or (),
                        candidate,
                        candidate_raw,
                    ) > allowed_conflicts:
                        continue
                return True
    return False


def _asr_fuzzy_phrase_match(
    q_words: tuple[str, ...],
    q_compact: str,
    row_segments: tuple[tuple[str, ...], ...],
    segment_lens: tuple[tuple[int, ...], ...],
    q_raw_words: tuple[str, ...] | None = None,
    q_has_diacritics: tuple[bool, ...] | None = None,
    raw_segments: tuple[tuple[str, ...], ...] | None = None,
) -> float:
    """Conservative one-edit ASR matching; short and numeric queries stay exact."""
    min_length = 7 if len(q_words) >= 2 else 5
    if len(q_compact) < min_length or any(char.isdigit() for char in q_compact):
        return 0.0
    word_count = len(q_words)
    window_sizes = (1, 2) if word_count == 1 else tuple(
        size for size in (word_count, word_count - 1, word_count + 1) if size >= 1
    )
    best = 0.0
    allowed_conflicts = _asr_conflicts_allowed(q_raw_words or ())
    query_length = len(q_compact)
    for segment_index, segment in enumerate(row_segments):
        token_lens = segment_lens[segment_index]
        for window_size in window_sizes:
            if len(segment) < window_size:
                continue
            for start in range(len(segment) - window_size + 1):
                stop = start + window_size
                candidate_length = sum(token_lens[start:stop])
                if abs(candidate_length - query_length) > 1:
                    continue
                # A one-edit match must agree at one edge (except for a
                # one-character insertion/deletion). This avoids most DP calls.
                if candidate_length == query_length and (
                    segment[start][0] != q_compact[0]
                    and segment[stop - 1][-1] != q_compact[-1]
                ):
                    continue
                if q_raw_words is not None and raw_segments is not None:
                    candidate_window = segment[start:stop]
                    raw_window = raw_segments[segment_index][start:stop]
                    if _asr_accent_conflicts(
                        q_raw_words,
                        q_has_diacritics or (),
                        candidate_window,
                        raw_window,
                    ) > allowed_conflicts:
                        continue
                candidate = "".join(segment[start:stop])
                cost = _unit_edit_cost(q_compact, candidate, max_edits=1)
                if cost <= 1.0:
                    best = max(best, 1.0 - cost / len(q_compact))
    return best


def _ocr_weighted_cost(a: str, b: str, max_cost: float, max_unit_edits: int = 1) -> float:
    """Confusion-weighted Levenshtein cost with early exit and unit-edit cap."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_cost + 1e-6:
        return max_cost + 1.0
    if a == b:
        return 0.0
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    prev_c = [float(i) for i in range(la + 1)]
    prev_u = list(range(la + 1))
    curr_c = [0.0] * (la + 1)
    curr_u = [0] * (la + 1)
    thresh = max_cost + 1e-6
    pairs = _CONFUSION_PAIRS

    for j in range(1, lb + 1):
        cb = b[j - 1]
        curr_c[0] = float(j)
        curr_u[0] = j
        min_row = float(j)
        for i in range(1, la + 1):
            ca = a[i - 1]
            if ca == cb:
                best_c = prev_c[i - 1]
                best_u = prev_u[i - 1]
            elif (ca, cb) in pairs:
                best_c = prev_c[i - 1] + 0.30
                best_u = prev_u[i - 1]
            else:
                best_c = prev_c[i - 1] + 1.0
                best_u = prev_u[i - 1] + 1

            ins_c = curr_c[i - 1] + 1.0
            ins_u = curr_u[i - 1] + 1
            if ins_c < best_c or (ins_c == best_c and ins_u < best_u):
                best_c = ins_c
                best_u = ins_u

            del_c = prev_c[i] + 1.0
            del_u = prev_u[i] + 1
            if del_c < best_c or (del_c == best_c and del_u < best_u):
                best_c = del_c
                best_u = del_u

            curr_c[i] = best_c
            curr_u[i] = best_u
            if best_c < min_row:
                min_row = best_c
        if min_row > thresh:
            return max_cost + 1.0
        prev_c, curr_c = curr_c, prev_c
        prev_u, curr_u = curr_u, prev_u

    final_c, final_u = prev_c[la], prev_u[la]
    if final_c > thresh or final_u > max_unit_edits:
        return max_cost + 1.0
    return final_c


def _make_trigrams(compact: str) -> tuple[str, ...]:
    if len(compact) < 3:
        return ()
    return tuple({compact[i : i + 3] for i in range(len(compact) - 2)})


def _fuzzy_budget(q_len: int) -> tuple[float, int]:
    """Return (max_weighted_cost, max_unit_edits) for a compact query of length q_len."""
    if q_len <= 3:
        return (0.0, 0)
    if q_len == 4:
        return (0.35, 0)
    max_cost = min(1.60, max(0.60, 0.20 * q_len))
    return (max_cost, 1)


def _fuzzy_phrase_match(
    q_words: tuple[str, ...],
    q_compact: str,
    q_trigrams: tuple[str, ...],
    row_segments: tuple[tuple[str, ...], ...],
    row_compact: str,
) -> float:
    q_len = len(q_compact)
    if not row_segments or q_len < 4:
        return 0.0
    max_cost, max_unit = _fuzzy_budget(q_len)
    if q_len >= 5 and q_trigrams and not any(tg in row_compact for tg in q_trigrams):
        pass  # Allow full segment check if confusion altered trigrams (e.g. 'koanken' vs 'hoankiem')

    k = len(q_words)
    window_sizes = (1, 2) if k == 1 else tuple(w for w in (k, k - 1, k + 1) if w >= 1)
    best = 0.0
    max_len_diff = int(max_cost) + 1
    q_mask = _conf_mask(q_compact)

    for seg_tokens in row_segments:
        n_tok = len(seg_tokens)
        for w_sz in window_sizes:
            if n_tok < w_sz:
                continue
            for i in range(n_tok - w_sz + 1):
                win = seg_tokens[i : i + w_sz]
                win_compact = "".join(win)
                if abs(len(win_compact) - q_len) > max_len_diff:
                    continue
                if q_len == 4 and len(win_compact) != 4:
                    continue
                missing = q_mask & ~_conf_mask(win_compact)
                if max_unit == 0:
                    if missing:
                        continue
                elif missing & (missing - 1):
                    continue
                cost = _ocr_weighted_cost(q_compact, win_compact, max_cost, max_unit_edits=max_unit)
                if cost <= max_cost + 1e-6:
                    score = max(0.05, 1.0 - (cost / max(q_len, 1)))
                    if score > best:
                        best = score
    return best


class CsvSearchEngine:
    """In-memory index over OCR / ASR / Metadata CSV files in the artifact directory."""

    def __init__(
        self,
        artifact_root: Path,
        source: str = "ocr",
        patterns: tuple[str, ...] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.artifact_root = Path(artifact_root)
        self.source = source
        self.patterns = patterns if patterns is not None else (("asr*.csv",) if source == "asr" else ("ocr*.csv",))
        self._mtimes: dict[str, float] = {}
        self.rows: list[dict] = []
        self.files_info: list[dict] = []
        self.rows_by_video: dict[str, list[int]] = {}
        self.vocab_postings: dict[str, list[int]] = {}
        self.asr_token_postings: dict[str, list[int]] = {}
        self.vocab_trigrams: dict[str, list[str]] = {}
        self.vocab_sig_trigrams: dict[str, list[str]] = {}
        self.vocab_by_len: dict[int, list[str]] = {}
        self.vocab_masks: dict[str, int] = {}
        self.row_sig_trigrams: dict[str, list[int]] = {}
        self._fuzzy_vocab_cache: dict[str, tuple[str, ...]] = {}
        self.reload_if_needed()

    def _discover_csv_files(self) -> list[Path]:
        files: list[Path] = []
        patterns = tuple(pattern.lower() for pattern in self.patterns)
        for directory in (self.artifact_root, self.artifact_root / "csv"):
            if directory.is_dir():
                files.extend(
                    path
                    for path in directory.iterdir()
                    if path.is_file() and any(fnmatch.fnmatchcase(path.name.lower(), pattern) for pattern in patterns)
                )
        return sorted(files, key=lambda path: (str(path.parent).lower(), path.name.lower()))

    def reload_if_needed(self, force: bool = False) -> None:
        with self._lock:
            self._reload_if_needed(force)

    def _reload_if_needed(self, force: bool = False) -> None:
        import csv

        csv_files = self._discover_csv_files()
        current_mtimes = {str(p): p.stat().st_mtime for p in csv_files if p.is_file()}
        if not force and current_mtimes == self._mtimes:
            return

        vid_pattern = re.compile(r"\b([LK]\d{2}_V\d{3})\b", re.IGNORECASE)
        loaded_rows: list[dict] = []
        info_list: list[dict] = []
        rows_by_video: dict[str, list[int]] = defaultdict(list)
        vocab_postings: dict[str, list[int]] = defaultdict(list)
        asr_token_postings: dict[str, list[int]] = defaultdict(list)
        row_sig_trigrams: dict[str, list[int]] = defaultdict(list)

        def _add_posting(key: str, ridx: int) -> None:
            plist = vocab_postings[key]
            if not plist or plist[-1] != ridx:
                plist.append(ridx)

        for path in csv_files:
            file_rows = 0
            try:
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    sample = handle.read(4096)
                    handle.seek(0)
                    delim = ","
                    if "\t" in sample and sample.count("\t") > sample.count(","):
                        delim = "\t"
                    elif ";" in sample and sample.count(";") > sample.count(","):
                        delim = ";"
                    reader = csv.reader(handle, delimiter=delim)
                    vid_col = -1
                    frame_col = -1
                    ts_col = -1
                    end_col = -1
                    ts_is_ms = False
                    end_is_ms = False
                    text_cols: list[int] = []
                    skip_cols: set[int] = set()
                    for row_idx, row in enumerate(reader):
                        if not row or not any(c.strip() for c in row):
                            continue
                        if row_idx == 0:
                            low = [c.strip().lower() for c in row]
                            if any(h in ("video_id", "video", "video_stem", "text", "ocr", "ocr_text", "transcript", "content") for h in low):
                                for ci, h in enumerate(low):
                                    if h in ("video_id", "video", "video_stem", "filename", "file") and vid_col < 0:
                                        vid_col = ci
                                    elif h in ("frame_id", "frame", "frame_idx") and frame_col < 0:
                                        frame_col = ci
                                    elif h in ("start", "timestamp", "time", "pts_time", "seconds", "timestamp_ms") and ts_col < 0:
                                        ts_col = ci
                                        ts_is_ms = h.endswith("_ms")
                                    elif h in ("end", "end_timestamp", "end_time", "end_ms") and end_col < 0:
                                        end_col = ci
                                        end_is_ms = h.endswith("_ms")
                                    elif h in ("text", "ocr", "ocr_text", "transcript", "content"):
                                        text_cols.append(ci)
                                    elif h in ("conf", "confidence", "score", "bbox", "box"):
                                        skip_cols.add(ci)
                                continue

                        video_stem = ""
                        if 0 <= vid_col < len(row):
                            video_stem = clean_video_name(row[vid_col]).upper()
                        if not video_stem or not vid_pattern.search(video_stem):
                            for cell in row:
                                m = vid_pattern.search(cell)
                                if m:
                                    video_stem = m.group(1).upper()
                                    break
                        if not video_stem:
                            continue

                        frame_id = None
                        if 0 <= frame_col < len(row):
                            try:
                                frame_id = int(float(row[frame_col].strip()))
                            except Exception:
                                pass

                        timestamp = None
                        if 0 <= ts_col < len(row):
                            try:
                                timestamp = float(row[ts_col].strip())
                                if ts_is_ms:
                                    timestamp /= 1000.0
                            except Exception:
                                pass

                        end_timestamp = None
                        if 0 <= end_col < len(row):
                            try:
                                end_timestamp = float(row[end_col].strip())
                                if end_is_ms:
                                    end_timestamp /= 1000.0
                            except Exception:
                                pass

                        if text_cols:
                            text_parts = [
                                row[ci].strip()
                                for ci in text_cols
                                if ci < len(row) and row[ci].strip() and row[ci].strip().upper() != video_stem
                            ]
                        else:
                            text_parts = [
                                c.strip()
                                for ci, c in enumerate(row)
                                if ci not in (vid_col, frame_col, ts_col, end_col) and ci not in skip_cols and c.strip() and c.strip().upper() != video_stem
                            ]
                        text_joined = " | ".join(text_parts)
                        if not text_joined:
                            continue
                        text_norm = strip_accents(text_joined)
                        if self.source == "asr":
                            text_ocr, row_segments, raw_segments = _asr_aligned_segments(text_joined)
                        else:
                            text_ocr, row_segments = _ocr_segments_and_text(text_norm)
                            raw_segments = ()
                        ridx = len(loaded_rows)
                        sig_segs_list: list[tuple[str, int, int]] = []
                        row_sig_seen: set[str] = set()
                        for seg_toks in row_segments:
                            seg_compact = "".join(seg_toks)
                            seg_sig = _conf_sig(seg_compact)
                            seg_mask = _conf_mask(seg_compact)
                            seg_len = len(seg_sig)
                            sig_segs_list.append((seg_sig, seg_mask, seg_len))
                            if seg_len >= 3:
                                for tj in range(seg_len - 2):
                                    stg = seg_sig[tj : tj + 3]
                                    if stg not in row_sig_seen:
                                        row_sig_seen.add(stg)
                                        row_sig_trigrams[stg].append(ridx)
                            nt = len(seg_toks)
                            for si, tok in enumerate(seg_toks):
                                if self.source == "asr":
                                    token_postings = asr_token_postings[tok]
                                    if not token_postings or token_postings[-1] != ridx:
                                        token_postings.append(ridx)
                                if len(tok) >= 2:
                                    _add_posting(tok, ridx)
                                if si + 1 < nt:
                                    w2 = tok + seg_toks[si + 1]
                                    if 3 <= len(w2) <= 24:
                                        _add_posting(w2, ridx)
                                    # Keep 3+ token postings short; longer fuzzy phrases use the
                                    # row signature index below instead of expanding every span.
                                    if len(w2) <= 6:
                                        acc = w2
                                        sj = si + 2
                                        while sj < nt and len(acc) < 6:
                                            acc += seg_toks[sj]
                                            sj += 1
                                            if 4 <= len(acc) <= 6:
                                                _add_posting(acc, ridx)
                        loaded_rows.append({
                            "video_stem": video_stem,
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "end_timestamp": end_timestamp,
                            "text": text_joined,
                            "text_low": text_joined.lower(),
                            "text_norm": text_norm,
                            "text_ocr": text_ocr,
                            "text_compact": text_ocr.replace(" ", ""),
                            "segments": row_segments,
                            "segment_lens": tuple(
                                tuple(len(token) for token in segment)
                                for segment in row_segments
                            ),
                            "raw_segments": raw_segments,
                            "sig_segs": tuple(sig_segs_list),
                            "file": path.name,
                        })
                        rows_by_video[video_stem].append(ridx)
                        file_rows += 1
            except Exception as exc:
                logger.warning("Failed reading CSV %s: %s", path, exc)
            info_list.append({"name": path.name, "rows": file_rows})

        vocab_trigrams: dict[str, list[str]] = defaultdict(list)
        vocab_sig_trigrams: dict[str, list[str]] = defaultdict(list)
        vocab_by_len: dict[int, list[str]] = defaultdict(list)
        vocab_masks: dict[str, int] = {}
        for vkey in vocab_postings:
            vocab_masks[vkey] = _conf_mask(vkey)
            vocab_by_len[len(vkey)].append(vkey)
            if len(vkey) >= 3:
                for tg in _make_trigrams(vkey):
                    vocab_trigrams[tg].append(vkey)
                for stg in _make_trigrams(_conf_sig(vkey)):
                    vocab_sig_trigrams[stg].append(vkey)

        self.rows = loaded_rows
        self.files_info = info_list
        self.rows_by_video = dict(rows_by_video)
        self.vocab_postings = dict(vocab_postings)
        self.asr_token_postings = dict(asr_token_postings)
        self.vocab_trigrams = dict(vocab_trigrams)
        self.vocab_sig_trigrams = dict(vocab_sig_trigrams)
        self.vocab_by_len = dict(vocab_by_len)
        self.vocab_masks = vocab_masks
        self.row_sig_trigrams = dict(row_sig_trigrams)
        self._fuzzy_vocab_cache = {}
        self._mtimes = current_mtimes

    def status(self) -> dict:
        with self._lock:
            self.reload_if_needed()
            unique_vids = len({r["video_stem"] for r in self.rows})
            return {
                "source": self.source,
                "total_rows": len(self.rows),
                "unique_videos": unique_vids,
                "files": self.files_info,
            }

    def _candidate_row_indices(
        self,
        terms: list[tuple],
        allowed_upper: set[str] | None,
        fuzzy: bool,
    ) -> list[int]:
        if allowed_upper is not None:
            allowed_rows: list[int] = []
            for stem in allowed_upper:
                allowed_rows.extend(self.rows_by_video.get(stem, ()))
            if len(allowed_rows) <= 8000:
                allowed_rows.sort()
                return allowed_rows
            allowed_row_set: set[int] | None = set(allowed_rows)
        else:
            allowed_row_set = None

        cand_set: set[int] = set()
        vocab_masks = self.vocab_masks
        for q_low, q_norm, q_ocr, q_words, q_compact, q_tri, is_num, q_raw_words, q_has_diacritics in terms:
            # 1) Exact / substring matches via trigram-indexed or direct vocab keys
            if self.source == "asr" and len(q_words) >= 2:
                posting_lists = [self.asr_token_postings.get(word, ()) for word in q_words]
                if all(posting_lists):
                    posting_lists.sort(key=len)
                    exact_rows = set(posting_lists[0])
                    for postings in posting_lists[1:]:
                        exact_rows.intersection_update(postings)
                        if not exact_rows:
                            break
                    cand_set.update(exact_rows)
            elif q_compact:
                if q_compact in self.vocab_postings:
                    cand_set.update(self.vocab_postings[q_compact])
                if len(q_compact) >= 3:
                    for vk in self.vocab_trigrams.get(q_compact[:3], ()):
                        if q_compact in vk:
                            cand_set.update(self.vocab_postings[vk])
                else:
                    for vk, plist in self.vocab_postings.items():
                        if q_compact in vk:
                            cand_set.update(plist)
            if self.source != "asr" and len(q_words) > 1:
                for qw in q_words:
                    if len(qw) >= 3 and qw in self.vocab_postings:
                        cand_set.update(self.vocab_postings[qw])

            # 2) Confusion-weighted fuzzy vocabulary expansion via canonical signature trigrams + cache
            asr_fuzzy_min = 7 if len(q_words) >= 2 else 5
            if fuzzy and self.source == "asr" and not is_num and len(q_compact) >= asr_fuzzy_min:
                q_len = len(q_compact)
                q_mask = _conf_mask(q_compact)
                q_sig = _conf_sig(q_compact)
                cache_key = f"asr:{q_compact}"
                matched_vks = self._fuzzy_vocab_cache.get(cache_key)
                if matched_vks is None:
                    vocab_pool: set[str] = set()
                    if q_len >= 6:
                        for stg in _make_trigrams(q_sig):
                            for vk in self.vocab_sig_trigrams.get(stg, ()):
                                if abs(len(vk) - q_len) <= 1:
                                    vocab_pool.add(vk)
                    else:  # A one-edit 5-character match may share no full trigram.
                        q_sig_bgs = tuple({q_sig[i : i + 2] for i in range(len(q_sig) - 1)})
                        for cand_len in range(q_len - 1, q_len + 2):
                            for vk in self.vocab_by_len.get(cand_len, ()):
                                vk_sig = _conf_sig(vk)
                                if any(bg in vk_sig for bg in q_sig_bgs):
                                    vocab_pool.add(vk)
                    matched_vks = tuple(
                        vk
                        for vk in vocab_pool
                        if _unit_edit_cost(q_compact, vk, max_edits=1) <= 1.0
                    )
                    if len(self._fuzzy_vocab_cache) >= 512:
                        self._fuzzy_vocab_cache.clear()
                    self._fuzzy_vocab_cache[cache_key] = matched_vks
                for vk in matched_vks:
                    cand_set.update(self.vocab_postings[vk])

                # 3) Multi-token ASR recall: one edit guarantees that at least
                # one half of the canonical query signature remains intact.
                if len(q_words) >= 2 and q_len >= 12:
                    parts = (q_sig[: q_len // 2], q_sig[q_len // 2 :])
                    row_sig_tris = self.row_sig_trigrams
                    for sub in parts:
                        tris = sorted(_make_trigrams(sub), key=lambda tg: len(row_sig_tris.get(tg, ())))
                        if not tris:
                            continue
                        sub_rows = set(row_sig_tris.get(tris[0], ()))
                        for tg in tris[1:]:
                            if not sub_rows:
                                break
                            sub_rows.intersection_update(row_sig_tris.get(tg, ()))
                        if allowed_row_set is not None:
                            sub_rows.intersection_update(allowed_row_set)
                        for idx in sub_rows:
                            if idx in cand_set:
                                continue
                            for sp, sm, slen in self.rows[idx]["sig_segs"]:
                                if slen >= q_len - 1 and sub in sp:
                                    missing = q_mask & ~sm
                                    if (missing & (missing - 1)) == 0:
                                        cand_set.add(idx)
                                        break

            # OCR confusion-weighted fuzzy vocabulary expansion.
            elif fuzzy and self.source != "asr" and not is_num and len(q_compact) >= 4:
                q_len = len(q_compact)
                max_cost, max_unit = _fuzzy_budget(q_len)
                q_mask = _conf_mask(q_compact)
                q_sig = _conf_sig(q_compact)
                min_vk_len = 4 if q_len == 4 else max(4, q_len - (int(max_cost) + 1))
                max_vk_len = 4 if q_len == 4 else q_len + (int(max_cost) + 1)

                matched_vks = self._fuzzy_vocab_cache.get(q_compact)
                if matched_vks is None:
                    q_sig_tris = _make_trigrams(q_sig)
                    tg_pool: set[str] = set()
                    for stg in q_sig_tris:
                        for vk in self.vocab_sig_trigrams.get(stg, ()):
                            if min_vk_len <= len(vk) <= max_vk_len:
                                missing = q_mask & ~vocab_masks[vk]
                                if (missing == 0) if max_unit == 0 else ((missing & (missing - 1)) == 0):
                                    tg_pool.add(vk)
                    if q_len == 5:
                        q_sig_bgs = tuple({q_sig[i : i + 2] for i in range(4)})
                        for cand_l in range(min_vk_len, max_vk_len + 1):
                            for vk in self.vocab_by_len.get(cand_l, ()):
                                missing = q_mask & ~vocab_masks[vk]
                                if (missing == 0) if max_unit == 0 else ((missing & (missing - 1)) == 0):
                                    vk_sig = _conf_sig(vk)
                                    if any(bg in vk_sig for bg in q_sig_bgs):
                                        tg_pool.add(vk)
                    matched_list = [
                        vk
                        for vk in tg_pool
                        if _ocr_weighted_cost(q_compact, vk, max_cost, max_unit_edits=max_unit) <= max_cost + 1e-6
                    ]
                    matched_vks = tuple(matched_list)
                    if len(self._fuzzy_vocab_cache) >= 512:
                        self._fuzzy_vocab_cache.clear()
                    self._fuzzy_vocab_cache[q_compact] = matched_vks
                for vk in matched_vks:
                    cand_set.update(self.vocab_postings[vk])

                # 3) Multi-token window recall (w >= 3): pigeonhole canonical signature parts over segment index
                if len(q_words) >= 2 and (q_len == 4 or q_len >= 6):
                    parts = (q_sig,) if max_unit == 0 else (q_sig[: q_len // 2], q_sig[q_len // 2 :])
                    row_sig_tris = self.row_sig_trigrams
                    for sub in parts:
                        tris = sorted(_make_trigrams(sub), key=lambda tg: len(row_sig_tris.get(tg, ())))
                        if not tris:
                            continue
                        sub_rows = set(row_sig_tris.get(tris[0], ()))
                        for tg in tris[1:]:
                            if not sub_rows:
                                break
                            sub_rows.intersection_update(row_sig_tris.get(tg, ()))
                        if allowed_row_set is not None:
                            sub_rows.intersection_update(allowed_row_set)
                        for idx in sub_rows:
                            if idx in cand_set:
                                continue
                            for sp, sm, slen in self.rows[idx]["sig_segs"]:
                                if slen >= min_vk_len and sub in sp:
                                    missing = q_mask & ~sm
                                    if (missing == 0) if max_unit == 0 else ((missing & (missing - 1)) == 0):
                                        cand_set.add(idx)
                                        break

        if allowed_row_set is not None:
            cand_set.intersection_update(allowed_row_set)
        return sorted(cand_set)

    def search(
        self,
        query: str,
        limit: int = 80,
        allowed_videos: set[str] | None = None,
        fuzzy: bool = True,
    ) -> dict:
        with self._lock:
            return self._search_unlocked(query, limit, allowed_videos, fuzzy)

    def _search_unlocked(
        self,
        query: str,
        limit: int = 80,
        allowed_videos: set[str] | None = None,
        fuzzy: bool = True,
    ) -> dict:
        self.reload_if_needed()
        q_raw = query.strip()
        if not q_raw:
            return {"source": self.source, "query": "", "video_ids": [], "matches": [], "total_rows": len(self.rows), "files": self.files_info}

        # Split by comma or semicolon into multiple keywords
        terms_raw = [t.strip() for t in q_raw.replace(";", ",").split(",") if t.strip()]
        terms = []
        for t in terms_raw:
            t_low = t.lower()
            t_norm = strip_accents(t)
            if self.source == "asr":
                t_ocr, query_segments, raw_query_segments = _asr_aligned_segments(t)
                q_words = tuple(token for segment in query_segments for token in segment)
            else:
                t_ocr = ocr_canonical(t)
                q_words = tuple(t_ocr.split())
                raw_query_segments = ()
            q_compact = "".join(q_words)
            is_num = (
                any(char.isdigit() for char in t_norm)
                if self.source == "asr"
                else t_norm.replace(" ", "").isdigit()
            )
            q_raw_words = tuple(token for segment in raw_query_segments for token in segment)
            q_has_diacritics = tuple(_has_diacritics(word) for word in q_raw_words)
            terms.append(
                (
                    t_low,
                    t_norm,
                    t_ocr,
                    q_words,
                    q_compact,
                    _make_trigrams(q_compact),
                    is_num,
                    q_raw_words,
                    q_has_diacritics,
                )
            )

        video_term_hits: dict[str, set[int]] = defaultdict(set)
        video_best_tier: dict[str, int] = defaultdict(int)
        video_score_sums: dict[str, float] = defaultdict(float)
        matched_snippets: list[dict] = []
        raw_cap = max(limit * 6, 360)
        allowed_upper = (
            {clean_video_name(v).upper() for v in allowed_videos}
            if allowed_videos is not None
            else None
        )

        cand_indices = self._candidate_row_indices(terms, allowed_upper, fuzzy)

        for ridx in cand_indices:
            r = self.rows[ridx]
            stem = r["video_stem"]
            t_low = r["text_low"]
            t_norm = r["text_norm"]
            t_ocr = r["text_ocr"]
            t_compact = r["text_compact"]
            t_segments = r["segments"]
            segment_lens = r["segment_lens"]
            raw_segments = r["raw_segments"]

            hit_idxs: list[int] = []
            row_best_tier = 0
            row_score = 0.0
            for ti, (
                q_low,
                q_norm,
                q_ocr,
                q_words,
                q_compact,
                q_tri,
                is_num,
                q_raw_words,
                q_has_diacritics,
            ) in enumerate(terms):
                if self.source == "asr" and q_low in t_low:
                    # Preserve accents as the strongest evidence. A raw word-boundary
                    # phrase outranks a mere accented substring.
                    is_boundary = bool(
                        re.search(rf"(?<!\w){re.escape(q_low)}(?!\w)", t_low)
                    )
                    tier = 5 if is_boundary else 4
                    hit_idxs.append(ti)
                    row_best_tier = max(row_best_tier, tier)
                    row_score += float(tier)
                elif self.source == "asr" and _asr_contiguous_accent_match(
                    q_words, q_raw_words, q_has_diacritics, t_segments, raw_segments
                ):
                    hit_idxs.append(ti)
                    row_best_tier = max(row_best_tier, 3)
                    row_score += 3.0
                elif self.source != "asr" and ((q_low in t_low) or bool(q_norm and q_norm in t_norm)):
                    # Keep the established OCR ranking behavior unchanged.
                    hit_idxs.append(ti)
                    is_boundary = bool(q_ocr and f" {q_ocr} " in f" {t_norm} ")
                    tier = 4 if is_boundary else 3
                    row_best_tier = max(row_best_tier, tier)
                    row_score += float(tier)
                elif not is_num and q_ocr and q_ocr in t_ocr and self.source != "asr":
                    hit_idxs.append(ti)
                    row_best_tier = max(row_best_tier, 2)
                    row_score += 2.0
                elif self.source == "asr" and len(q_words) >= 2 and _asr_tokens_in_order(
                    q_words,
                    t_segments,
                    q_raw_words=q_raw_words,
                    q_has_diacritics=q_has_diacritics,
                    raw_segments=raw_segments,
                ):
                    hit_idxs.append(ti)
                    row_best_tier = max(row_best_tier, 2)
                    row_score += 1.8
                elif fuzzy and not is_num:
                    f_score = (
                        _asr_fuzzy_phrase_match(
                            q_words,
                            q_compact,
                            t_segments,
                            segment_lens,
                            q_raw_words=q_raw_words,
                            q_has_diacritics=q_has_diacritics,
                            raw_segments=raw_segments,
                        )
                        if self.source == "asr"
                        else _fuzzy_phrase_match(q_words, q_compact, q_tri, t_segments, t_compact)
                    )
                    if f_score > 0.0:
                        hit_idxs.append(ti)
                        row_best_tier = max(row_best_tier, 1)
                        row_score += f_score

            if hit_idxs:
                video_term_hits[stem].update(hit_idxs)
                if row_best_tier > video_best_tier[stem]:
                    video_best_tier[stem] = row_best_tier
                video_score_sums[stem] += row_score
                if len(matched_snippets) < raw_cap or row_best_tier >= 2:
                    frame_id = r["frame_id"]
                    ts_val = r["timestamp"]
                    end_ts_val = r["end_timestamp"]
                    match_type = "exact" if row_best_tier >= 3 else ("ocr" if row_best_tier == 2 else "fuzzy")
                    matched_snippets.append({
                        "source": self.source,
                        "video_id": stem,
                        "frame_id": frame_id,
                        "timestamp": ts_val,
                        "end_timestamp": end_ts_val,
                        "timestamp_ms": round(ts_val * 1000) if ts_val is not None else None,
                        "end_ms": round(end_ts_val * 1000) if end_ts_val is not None else None,
                        "image_path": f"images/{stem}/frame_{frame_id}.webp" if frame_id is not None else None,
                        "text": r["text"][:220],
                        "file": r["file"],
                        "matched_terms": len(hit_idxs),
                        "match_tier": row_best_tier,
                        "match_score": round(row_score, 3),
                        "match_type": match_type,
                    })

        # Prefer videos containing ALL terms, then exact > OCR-normalized > fuzzy, then hit count
        all_term_videos = [v for v, hits in video_term_hits.items() if len(hits) == len(terms)]
        target_videos = all_term_videos if all_term_videos else list(video_term_hits.keys())
        target_videos.sort(
            key=lambda v: (len(video_term_hits[v]), video_best_tier[v], video_score_sums[v], v),
            reverse=True,
        )

        target_set = set(target_videos)
        filtered_snippets = [m for m in matched_snippets if m["video_id"] in target_set]
        filtered_snippets.sort(
            key=lambda m: (m["matched_terms"], m["match_tier"], m["match_score"]),
            reverse=True,
        )
        filtered_snippets = filtered_snippets[:limit]

        return {
            "source": self.source,
            "query": q_raw,
            "video_ids": target_videos,
            "video_count": len(target_videos),
            "all_terms_matched": bool(all_term_videos),
            "matches": filtered_snippets,
            "total_rows": len(self.rows),
            "files": self.files_info,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build local memory-mapped index from Spatial-KIS artifact")
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()
    idx = LocalMemmapIndex(args.artifact)
    # Warm up global region 0
    _ = idx._region_tensor_fp32(0)
    print(json.dumps({"ok": True, "count": idx.count, "dim": idx.dim, "index_dir": str(idx.index_dir)}))


if __name__ == "__main__":
    main()
