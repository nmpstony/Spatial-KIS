from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class KeyframeRecord:
    keyframe_id: str
    video_id: str
    shot_id: int
    frame_id: int
    fps: float
    timestamp: float
    width: int
    height: int
    sharpness: float
    image_path: str
    embedding_shard: str
    embedding_offset: int

    def as_dict(self) -> dict:
        return asdict(self)


def config_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactWriter:
    def __init__(self, root: Path, run_id: str, config: dict):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(exist_ok=True)
        (self.root / "embeddings").mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        state_path = self.root / ".run_state.json"
        requested_hash = config_hash(config)
        if state_path.exists():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if previous["config_hash"] != requested_hash:
                raise ValueError("Refusing to resume an artifact with a different preprocess config")
            run_id = previous["run_id"]
        else:
            state_path.write_text(json.dumps({"run_id": run_id, "config_hash": requested_hash}), encoding="utf-8")
        self.run_id = run_id
        self.config = config

    def append_manifest(self, record: KeyframeRecord) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    def write_video_embeddings(self, video_id: str, vectors: np.ndarray) -> str:
        # Each row has exactly global + nine regional embeddings.
        if vectors.ndim != 3 or vectors.shape[1] != 10:
            raise ValueError("Expected [keyframes, 10, dimension] embeddings")
        safe_name = video_id.replace("/", "__").replace("\\", "__")
        relative = f"embeddings/{safe_name}.npz"
        np.savez_compressed(self.root / relative, vectors=vectors.astype(np.float16))
        return relative

    def finalize(self, source_video_count: int) -> None:
        checksums = {"manifest.jsonl": file_checksum(self.manifest_path)} if self.manifest_path.exists() else {}
        for shard in sorted((self.root / "embeddings").glob("*.npz")):
            checksums[str(shard.relative_to(self.root)).replace("\\", "/")] = file_checksum(shard)
        metadata = {
            "format": 1,
            "run_id": self.run_id,
            "config": self.config,
            "config_hash": config_hash(self.config),
            "source_video_count": source_video_count,
            "checksums": checksums,
        }
        (self.root / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def iter_manifest(root: Path) -> Iterator[KeyframeRecord]:
    path = root / "manifest.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield KeyframeRecord(**json.loads(line))


def stable_row_id(keyframe_id: str, region_id: int) -> int:
    # Positive signed 63-bit id for Milvus. Collision is detected by unique source IDs on import.
    value = hashlib.blake2b(f"{keyframe_id}:{region_id}".encode(), digest_size=8).digest()
    return int.from_bytes(value, "big") & ((1 << 63) - 1)
