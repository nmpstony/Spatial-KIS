import json
import tempfile
from pathlib import Path
import numpy as np
from spatial_kis.local_index import LocalMemmapIndex, clean_video_name


def test_clean_video_name_strips_extension_and_directory():
    assert clean_video_name("L01_V001.mp4") == "L01_V001"
    assert clean_video_name("L01/L01_V002.MP4") == "L01_V002"
    assert clean_video_name("L02_V010") == "L02_V010"


def test_local_memmap_index_builds_and_searches():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "embeddings").mkdir()
        vecs = np.zeros((2, 10, 4), dtype=np.float16)
        # Keyframe 0, region 0 matches [1, 0, 0, 0]
        vecs[0, 0] = [1.0, 0.0, 0.0, 0.0]
        # Keyframe 1, region 0 matches [0, 1, 0, 0]
        vecs[1, 0] = [0.0, 1.0, 0.0, 0.0]
        np.savez_compressed(root / "embeddings" / "L01_V001.mp4.npz", vectors=vecs)

        manifest = [
            {
                "keyframe_id": "L01_V001.mp4:0",
                "video_id": "L01_V001.mp4",
                "shot_id": 0,
                "frame_id": 0,
                "fps": 25.0,
                "timestamp": 1.0,
                "width": 1280,
                "height": 720,
                "sharpness": 100.0,
                "image_path": "images/L01_V001/frame_0.webp",
                "embedding_shard": "embeddings/L01_V001.mp4.npz",
                "embedding_offset": 0,
            },
            {
                "keyframe_id": "L01_V001.mp4:100",
                "video_id": "L01_V001.mp4",
                "shot_id": 1,
                "frame_id": 100,
                "fps": 25.0,
                "timestamp": 5.0,
                "width": 1280,
                "height": 720,
                "sharpness": 100.0,
                "image_path": "images/L01_V001/frame_100.webp",
                "embedding_shard": "embeddings/L01_V001.mp4.npz",
                "embedding_offset": 1,
            },
        ]
        (root / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")

        idx = LocalMemmapIndex(root)
        res = idx.search_conditions(
            evaluated_conditions=[((0,), [[1.0, 0.0, 0.0, 0.0]], 1.0, 1)],
            top_k=10,
            result_limit=2,
            diversify=False,
        )
        assert res["results"][0]["keyframe_id"] == "L01_V001.mp4:0"
        assert res["results"][0]["video_stem"] == "L01_V001"
        assert res["results"][0]["timestamp_ms"] == 1000

        # Test temporal search (Stage 1 @ t=1.0s followed by Stage 2 @ t=5.0s)
        res_temp = idx.search_conditions(
            evaluated_conditions=[
                ((0,), [[1.0, 0.0, 0.0, 0.0]], 1.0, 1),
                ((0,), [[0.0, 1.0, 0.0, 0.0]], 1.0, 2),
            ],
            top_k=10,
            result_limit=2,
            diversify=False,
            max_temporal_gap=10.0,
        )
        assert res_temp["results"][0]["keyframe_id"] == "L01_V001.mp4:0"
        assert res_temp["results"][0]["next_match"]["timestamp_ms"] == 5000
