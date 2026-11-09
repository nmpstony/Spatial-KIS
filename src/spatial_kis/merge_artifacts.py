from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

from .artifact import ArtifactWriter, iter_manifest
from .verify_artifact import verify


def merge(inputs: list[Path], output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new empty directory")
    runs = [json.loads((item / "run.json").read_text(encoding="utf-8")) for item in inputs]
    if len({run["config_hash"] for run in runs}) != 1:
        raise ValueError("Cannot merge artifacts created with different preprocess configs")
    writer = ArtifactWriter(output, uuid.uuid4().hex, runs[0]["config"])
    keyframes: set[str] = set()
    for source, run in zip(inputs, runs):
        verify(source)
        for folder in ("images", "embeddings"):
            shutil.copytree(source / folder, output / folder, dirs_exist_ok=True)
        for record in iter_manifest(source):
            if record.keyframe_id in keyframes:
                raise ValueError(f"Duplicate keyframe: {record.keyframe_id}")
            keyframes.add(record.keyframe_id)
            writer.append_manifest(record)
    writer.finalize(sum(run["source_video_count"] for run in runs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently generated Kaggle artifact chunks")
    parser.add_argument("--inputs", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge(args.inputs, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
