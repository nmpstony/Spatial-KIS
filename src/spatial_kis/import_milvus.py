from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymilvus import DataType, MilvusClient

from .artifact import iter_manifest, stable_row_id
from .config import load_profiles
from .verify_artifact import verify


def collection_name(run_id: str, profile: str) -> str:
    return f"spatial_kis_{run_id[:12]}_{profile.replace('-', '_')}"


def estimate_gpu_bytes(row_count: int, dimension: int) -> int:
    # CAGRA documentation estimates roughly 1.8x original data; include 15% operational headroom.
    return int(row_count * dimension * 4 * 1.8 * 1.15)


def import_artifact(artifact: Path, profile_name: str, profiles_path: Path, uri: str, skip_verify: bool = False) -> str:
    summary = {"vectors": 0} if skip_verify else verify(artifact)
    run = json.loads((artifact / "run.json").read_text(encoding="utf-8"))
    profile = load_profiles(profiles_path)[profile_name]
    records = list(iter_manifest(artifact))
    first = np.load(artifact / records[0].embedding_shard)["vectors"]
    dimension = int(first.shape[-1])
    row_count = len(records) * 10
    if summary["vectors"] == 0:
        summary["vectors"] = row_count
    if row_count > profile["max_rows"]:
        raise ValueError(f"{row_count} rows exceed profile limit")
    if profile["index_type"] == "GPU_CAGRA":
        estimate = estimate_gpu_bytes(row_count, dimension)
        safe = int(profile["gpu_vram_bytes"] * profile["safe_vram_fraction"])
        if estimate > safe:
            raise RuntimeError(f"Estimated GPU need {estimate:,} bytes exceeds safe limit {safe:,}; use portable-local")

    name = collection_name(run["run_id"], profile_name)
    client = MilvusClient(uri=uri)
    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("keyframe_id", DataType.VARCHAR, max_length=320)
    schema.add_field("video_id", DataType.VARCHAR, max_length=256)
    schema.add_field("shot_id", DataType.INT64)
    schema.add_field("frame_id", DataType.INT64)
    schema.add_field("timestamp", DataType.DOUBLE)
    schema.add_field("region_id", DataType.INT8)
    schema.add_field("image_path", DataType.VARCHAR, max_length=384)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
    
    # Create collection without index_params to avoid concurrent index building during bulk insertion
    client.create_collection(name, schema=schema)

    by_shard: dict[str, list] = defaultdict(list)
    for record in records:
        by_shard[record.embedding_shard].append(record)

    batch: list[dict] = []
    seen: set[int] = set()
    inserted_count = 0
    total_shards = len(by_shard)

    print(f"Starting import into Milvus collection '{name}': {row_count} rows from {total_shards} shards...", flush=True)

    for shard_idx, (shard_name, shard_records) in enumerate(by_shard.items(), 1):
        vectors = np.load(artifact / shard_name)["vectors"]
        for record in shard_records:
            for region_id, vector in enumerate(vectors[record.embedding_offset].astype(np.float32)):
                row_id = stable_row_id(record.keyframe_id, region_id)
                if row_id in seen:
                    raise RuntimeError(f"Hash collision at {record.keyframe_id}")
                seen.add(row_id)
                batch.append({"id": row_id, "keyframe_id": record.keyframe_id, "video_id": record.video_id,
                              "shot_id": record.shot_id, "frame_id": record.frame_id, "timestamp": record.timestamp,
                              "region_id": region_id, "image_path": record.image_path, "embedding": vector.tolist()})
            if len(batch) >= 5000:
                client.insert(name, batch)
                inserted_count += len(batch)
                batch.clear()
        if shard_idx % 50 == 0 or shard_idx == total_shards:
            print(f"  Shard [{shard_idx}/{total_shards}] - Inserted {inserted_count:,}/{row_count:,} rows ({inserted_count*100/row_count:.1f}%)", flush=True)

    if batch:
        client.insert(name, batch)
        inserted_count += len(batch)
        batch.clear()

    print(f"Flushing collection '{name}'...", flush=True)
    client.flush(name)
    actual = client.get_collection_stats(name)["row_count"]
    if int(actual) != row_count:
        raise RuntimeError(f"Import mismatch: expected {row_count}, got {actual}")

    print(f"Building {profile_name} index on '{name}'...", flush=True)
    index_params = client.prepare_index_params()
    index_params.add_index("embedding", index_type=profile["index_type"], metric_type=profile["metric_type"], params=profile["params"])
    client.create_index(name, index_params)

    print(f"Loading collection '{name}' into memory...", flush=True)
    client.load_collection(name)

    state = {"active_collection": name, "run_id": run["run_id"], "profile": profile_name, "rows": row_count,
             "vectors": summary["vectors"], "metric": "IP", "search_params": profile["search_params"]}
    state_path = artifact / "active_index.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"SUCCESS! Imported and indexed {row_count:,} rows in '{name}'. active_index.json written.", flush=True)
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--profile", choices=("portable-local", "quality-gpu"), required=True)
    parser.add_argument("--profiles", type=Path, default=Path("config/index_profiles.json"))
    parser.add_argument("--uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    print(import_artifact(args.artifact, args.profile, args.profiles, args.uri, args.skip_verify))


if __name__ == "__main__":
    main()
