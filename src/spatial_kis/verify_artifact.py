from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifact import file_checksum, iter_manifest


def verify(root: Path) -> dict:
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    records = list(iter_manifest(root))
    if not records:
        raise ValueError("Artifact has no retained keyframes")
    by_shard: dict[str, list] = defaultdict(list)
    for record in records:
        if record.fps <= 0 or record.timestamp < 0:
            raise ValueError(f"Invalid time metadata: {record.keyframe_id}")
        if not (root / record.image_path).is_file():
            raise FileNotFoundError(record.image_path)
        by_shard[record.embedding_shard].append(record)
    rows = 0
    for shard_name, shard_records in by_shard.items():
        shard = root / shard_name
        if not shard.is_file():
            raise FileNotFoundError(shard_name)
        vectors = np.load(shard)["vectors"].astype(np.float32)
        if vectors.ndim != 3 or vectors.shape[1] != 10:
            raise ValueError(f"{shard_name} does not have 10 vectors/keyframe")
        for record in shard_records:
            vector = vectors[record.embedding_offset]
            if not np.allclose(np.linalg.norm(vector, axis=1), 1.0, atol=2e-2):
                raise ValueError(f"Unnormalized embedding: {record.keyframe_id}")
        rows += len(shard_records) * 10
    for name, expected in run["checksums"].items():
        if file_checksum(root / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return {"keyframes": len(records), "vectors": rows, "shards": len(by_shard)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact), indent=2))


if __name__ == "__main__":
    main()
