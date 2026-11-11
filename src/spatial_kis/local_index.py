from __future__ import annotations

import json
import pickle
from collections import defaultdict
from functools import lru_cache
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


class LocalMemmapIndex:
    """Zero-Docker memory-mapped exact vector search engine for Spatial-KIS."""

    def __init__(self, artifact_root: Path) -> None:
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

    @lru_cache(maxsize=1)
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
        allowed: set[str] = set()
        for stem in self.by_video:
            clean_st = clean_video_name(stem)
            stem_up = clean_st.upper()
            if any(stem_up == tok or stem_up.startswith(tok + "_") or stem_up.startswith(tok) for tok in tokens):
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


def strip_accents(text: str) -> str:
    import unicodedata

    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").lower()


class CsvSearchEngine:
    """In-memory index over OCR / ASR / Metadata CSV files in the artifact directory."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)
        self._mtimes: dict[str, float] = {}
        self.rows: list[dict] = []
        self.files_info: list[dict] = []
        self.reload_if_needed()

    def _discover_csv_files(self) -> list[Path]:
        files: list[Path] = []
        if self.artifact_root.is_dir():
            files.extend(sorted(self.artifact_root.glob("*.csv")))
            csv_sub = self.artifact_root / "csv"
            if csv_sub.is_dir():
                files.extend(sorted(csv_sub.glob("*.csv")))
        return files

    def reload_if_needed(self, force: bool = False) -> None:
        import csv
        import re

        csv_files = self._discover_csv_files()
        current_mtimes = {str(p): p.stat().st_mtime for p in csv_files if p.is_file()}
        if not force and current_mtimes == self._mtimes:
            return

        vid_pattern = re.compile(r"\b([LK]\d{2}_V\d{3})\b", re.IGNORECASE)
        loaded_rows: list[dict] = []
        info_list: list[dict] = []

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
                    header: list[str] | None = None
                    vid_col = -1
                    frame_col = -1
                    ts_col = -1
                    for row_idx, row in enumerate(reader):
                        if not row or not any(c.strip() for c in row):
                            continue
                        if row_idx == 0:
                            low = [c.strip().lower() for c in row]
                            if any(h in ("video_id", "video", "video_stem", "text", "ocr", "ocr_text", "transcript", "content") for h in low):
                                header = low
                                for ci, h in enumerate(low):
                                    if h in ("video_id", "video", "video_stem", "filename", "file") and vid_col < 0:
                                        vid_col = ci
                                    elif h in ("frame_id", "frame", "frame_idx") and frame_col < 0:
                                        frame_col = ci
                                    elif h in ("timestamp", "time", "pts_time", "seconds", "timestamp_ms") and ts_col < 0:
                                        ts_col = ci
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
                            except Exception:
                                pass

                        text_parts = [
                            c.strip()
                            for ci, c in enumerate(row)
                            if ci not in (vid_col, frame_col, ts_col) and c.strip() and c.strip().upper() != video_stem
                        ]
                        text_joined = " | ".join(text_parts)
                        if not text_joined:
                            continue
                        loaded_rows.append({
                            "video_stem": video_stem,
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "text": text_joined,
                            "text_low": text_joined.lower(),
                            "text_norm": strip_accents(text_joined),
                            "file": path.name,
                        })
                        file_rows += 1
            except Exception:
                pass
            info_list.append({"name": path.name, "rows": file_rows})

        self.rows = loaded_rows
        self.files_info = info_list
        self._mtimes = current_mtimes

    def status(self) -> dict:
        self.reload_if_needed()
        unique_vids = len({r["video_stem"] for r in self.rows})
        return {
            "total_rows": len(self.rows),
            "unique_videos": unique_vids,
            "files": self.files_info,
        }

    def search(self, query: str, limit: int = 80) -> dict:
        self.reload_if_needed()
        q_raw = query.strip()
        if not q_raw:
            return {"query": "", "video_ids": [], "matches": [], "total_rows": len(self.rows), "files": self.files_info}

        # Split by comma or semicolon into multiple keywords
        terms_raw = [t.strip() for t in q_raw.replace(";", ",").split(",") if t.strip()]
        terms = [(t.lower(), strip_accents(t)) for t in terms_raw]

        video_term_hits: dict[str, set[int]] = defaultdict(set)
        video_row_counts: dict[str, int] = defaultdict(int)
        matched_snippets: list[dict] = []

        for r in self.rows:
            t_low = r["text_low"]
            t_norm = r["text_norm"]
            hit_idxs = [
                ti for ti, (q_low, q_norm) in enumerate(terms)
                if q_low in t_low or (q_norm and q_norm in t_norm)
            ]
            if hit_idxs:
                stem = r["video_stem"]
                video_term_hits[stem].update(hit_idxs)
                video_row_counts[stem] += len(hit_idxs)
                if len(matched_snippets) < limit:
                    matched_snippets.append({
                        "video_id": stem,
                        "frame_id": r["frame_id"],
                        "timestamp": r["timestamp"],
                        "text": r["text"][:220],
                        "file": r["file"],
                        "matched_terms": len(hit_idxs),
                    })

        # If multiple terms were given, prefer videos that contain ALL terms (intersection across terms); fallback to most terms matched
        all_term_videos = [v for v, hits in video_term_hits.items() if len(hits) == len(terms)]
        target_videos = all_term_videos if all_term_videos else list(video_term_hits.keys())
        target_videos.sort(key=lambda v: (len(video_term_hits[v]), video_row_counts[v], v), reverse=True)

        target_set = set(target_videos)
        filtered_snippets = [m for m in matched_snippets if m["video_id"] in target_set][:limit]

        return {
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
