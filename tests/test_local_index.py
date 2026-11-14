import json
import tempfile
from pathlib import Path
import numpy as np
from spatial_kis.local_index import CsvSearchEngine, LocalMemmapIndex, clean_video_name


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

        # Test CSV search engine (with and without Vietnamese accents)
        csv_content = "video_id,frame_id,timestamp,text\nL01_V001,100,5.0,Vòng đua thành phố Tuyên Quang\nL22_V011,250,10.0,Bản tin thời sự buổi sáng\n"
        (root / "ocr.csv").write_text(csv_content, encoding="utf-8")
        csv_eng = CsvSearchEngine(root)
        hit_acc = csv_eng.search("Tuyên Quang")
        assert hit_acc["video_ids"] == ["L01_V001"]
        assert hit_acc["matches"][0]["image_path"] == "images/L01_V001/frame_100.webp"
        assert hit_acc["matches"][0]["timestamp_ms"] == 5000
        hit_no_acc = csv_eng.search("tuyen quang")
        assert hit_no_acc["video_ids"] == ["L01_V001"]
        assert csv_eng.search("tuyen quang", allowed_videos={"L22_V011"})["video_ids"] == []
        assert csv_eng.search("tuyen quang", allowed_videos={"L01_V001"})["video_ids"] == ["L01_V001"]

        # Test video_filter intersection
        allowed = idx.resolve_allowed_videos(["L01_V001"])
        assert allowed == {"L01_V001"}
        empty_allowed = idx.resolve_allowed_videos(["L99_V999"])
        res_filtered = idx.search_conditions(
            evaluated_conditions=[((0,), [[1.0, 0.0, 0.0, 0.0]], 1.0, 1)],
            top_k=10,
            result_limit=2,
            diversify=False,
            allowed_videos=empty_allowed,
        )
        assert res_filtered["results"] == []
