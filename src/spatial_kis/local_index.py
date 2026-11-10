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

    @lru_cache(maxsize=3)
    def _region_tensor_fp32(self, region_id: int) -> torch.Tensor:
        """Cache up to 3 regions in float32 RAM (~1.29 GB each) for sub-100ms BLAS matmul."""
        arr = np.array(self._mmaps[region_id], dtype=np.float32, copy=True)
        return torch.from_numpy(arr)

    def score_region(self, region_id: int, query_vectors: list[list[float]]) -> np.ndarray:
        """Return max inner-product score per keyframe across query_vectors for a single region."""
        q = torch.tensor(query_vectors, dtype=torch.float32).T  # [dim, V]
        mat = self._region_tensor_fp32(region_id)  # [N, dim]
        with torch.no_grad():
            scores = torch.matmul(mat, q)  # [N, V]
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

    def search_conditions(
        self,
        evaluated_conditions: list[tuple[tuple[int, ...], list[list[float]], float, int]],
        top_k: int,
        result_limit: int,
        diversify: bool,
        max_temporal_gap: float = 25.0,
    ) -> dict:
        """
        Full-corpus multi-condition & temporal search.
        Each item in evaluated_conditions is: (region_ids_tuple, query_vectors, weight, stage).
        stage=1: primary scene; stage=2: subsequent scene within max_temporal_gap seconds in the same video.
        """
        stage1 = [item for item in evaluated_conditions if item[3] <= 1]
        stage2 = [item for item in evaluated_conditions if item[3] >= 2]
        if not stage1:
            stage1, stage2 = stage2, []

        def combine_stage(conds: list[tuple[tuple[int, ...], list[list[float]], float, int]]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
            raw_scores_list: list[np.ndarray] = []
            norm_scores_list: list[np.ndarray] = []
            weights: list[float] = []
            primary_regions: np.ndarray | None = None
            for regions_tuple, q_vecs, weight, _ in conds:
                scores, regs = self.score_condition(regions_tuple, q_vecs)
                raw_scores_list.append(scores)
                norm_scores_list.append(np.clip((scores + 1.0) * 0.5, 1e-6, 1.0))
                weights.append(float(weight))
                if primary_regions is None:
                    primary_regions = regs
            assert primary_regions is not None
            if len(conds) == 1:
                combined = norm_scores_list[0]
            else:
                w_sum = sum(weights)
                denom = np.zeros(self.count, dtype=np.float32)
                for norm_s, w in zip(norm_scores_list, weights):
                    denom += w / norm_s
                combined = w_sum / denom
            return combined, primary_regions, raw_scores_list

        s1_combined, s1_regions, s1_raw = combine_stage(stage1)

        matched_next_idx: dict[int, int] = {}
        if stage2:
            s2_combined, _, _ = combine_stage(stage2)
            final_scores = np.zeros(self.count, dtype=np.float32)
            # Only evaluate temporal window on top Stage-1 candidates for <20ms speed
            cand_k = min(15000, self.count)
            top_s1_indices = np.argpartition(s1_combined, -cand_k)[-cand_k:]
            for idx_a in top_s1_indices:
                idx_a_int = int(idx_a)
                rec_a = self.records[idx_a_int]
                stem = rec_a["video_stem"]
                t_a = rec_a["timestamp"]
                v_idxs = self.by_video.get(stem, [])
                best_b = -1
                best_s2 = -1.0
                for idx_b in v_idxs:
                    if idx_b == idx_a_int:
                        continue
                    dt = self.records[idx_b]["timestamp"] - t_a
                    if dt > max_temporal_gap:
                        break
                    # Require strictly subsequent frame (at least 0.5s after Stage 1)
                    if dt >= 0.5:
                        val_s2 = float(s2_combined[idx_b])
                        if val_s2 > best_s2:
                            best_s2 = val_s2
                            best_b = idx_b
                if best_b >= 0:
                    s1_val = max(float(s1_combined[idx_a_int]), 1e-6)
                    s2_val = max(best_s2, 1e-6)
                    final_scores[idx_a_int] = 2.0 / (1.0 / s1_val + 1.0 / s2_val)
                    matched_next_idx[idx_a_int] = best_b
        else:
            final_scores = s1_combined

        k = min(max(top_k, result_limit * 4), self.count)
        top_indices = np.argpartition(final_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

        ranked: list[dict] = []
        for idx in top_indices:
            idx_int = int(idx)
            rec = self.records[idx_int]
            entry = {
                **rec,
                "score": float(final_scores[idx_int]),
                "region_id": int(s1_regions[idx_int]),
                "condition_scores": [float(arr[idx_int]) for arr in s1_raw],
            }
            if idx_int in matched_next_idx:
                next_rec = self.records[matched_next_idx[idx_int]]
                entry["next_match"] = {
                    "keyframe_id": next_rec["keyframe_id"],
                    "frame_id": next_rec["frame_id"],
                    "timestamp": next_rec["timestamp"],
                    "timestamp_ms": next_rec["timestamp_ms"],
                    "image_path": next_rec["image_path"],
                }
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
