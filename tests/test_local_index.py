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

        # Test OCR-tolerant & fuzzy matching for noisy Vietnamese OCR text
        noisy_csv = (
            "video_id,frame_id,timestamp,text,confidence\n"
            "L01_V001,100,5.0,Thành phố Sầm Sơn,0.98\n"
            "L23_V007,7679,307.16,Đích @ld Sam Sđn (Thonh Hoa),0.81\n"
            "L23_V022,7355,294.2,MUMG 70 Nom DIEN Vien Puui7s,0.77\n"
            "L23_V013,6348,253.92,Đà Nẵng - Jam Kỳ (Quang Nam),0.85\n"
            "L23_V009,7498,299.92,Vinh (Nghê An) -Dông Hql (euang Binh),0.72\n"
            "L23_V005,4008,160.32,koàn Kén (Hàl6),0.69\n"
            "L23_V002,5900,236.0,Vòng đua thònh phố San lo,0.75\n"
            "L23_V018,10,1.0,Đài Truyền Hình Thành Phố Hồ Chí Minh,0.95\n"
            "L23_V019,12,1.2,Đích ihonh hqa,0.70\n"
            "L23_V021,14,1.4,Tin tức ko cki mink,0.74\n"
            "L23_V023,16,1.6,Chặng đua Th anh Hqa,0.72\n"
            "L23_V024,18,1.8,Đích thank pko ko cki mink,0.71\n"
            "L23_V025,20,2.0,Tỉnh s on lx buổi sáng,0.68\n"
            "L23_V026,22,2.2,Tin ko ch i mink,0.73\n"
            "L02_V001,10,1.0,Bản tin be nhỏ tu yen,0.91\n"
        )
        (root / "ocr.csv").write_text(noisy_csv, encoding="utf-8")
        csv_eng.reload_if_needed(force=True)
        sam_son = csv_eng.search("Sầm Sơn")
        assert sam_son["video_ids"] == ["L01_V001", "L23_V007"]
        assert sam_son["matches"][0]["match_type"] == "exact"
        thanh_hoa_vids = csv_eng.search("Thanh Hóa")["video_ids"]
        assert "L23_V007" in thanh_hoa_vids
        assert "L23_V019" in thanh_hoa_vids  # Zero raw trigrams shared ('ihonhhqa' vs 'thanhhoa')
        assert "L23_V023" in thanh_hoa_vids  # 2-word query matching 3-token window ('Th anh Hqa')
        assert "L23_V018" not in thanh_hoa_vids  # Must NOT false-match 'Thành Phố'
        ho_chi_minh_vids = csv_eng.search("Ho Chi Minh")["video_ids"]
        assert "L23_V018" in ho_chi_minh_vids
        assert "L23_V021" in ho_chi_minh_vids  # 3-token confusion match ('ko cki mink') in unrestricted index
        assert "L23_V026" in ho_chi_minh_vids  # 4-token split match ('ko ch i mink') for 3-word query
        assert "L23_V024" in csv_eng.search("Thanh Pho Ho Chi Minh")["video_ids"]  # 5-token query/window
        assert "L23_V022" in csv_eng.search("Điện Biên")["video_ids"]
        assert "L23_V013" in csv_eng.search("Tam Kỳ")["video_ids"]
        assert "L23_V009" in csv_eng.search("Đồng Hới")["video_ids"]
        assert "L23_V005" in csv_eng.search("Hoàn Kiếm")["video_ids"]
        son_la_vids = csv_eng.search("Sơn La")["video_ids"]
        assert "L23_V002" in son_la_vids
        assert "L23_V025" in son_la_vids  # 5-char query across 3-token window ('s on lx')
        assert "L02_V001" in csv_eng.search("tuyen")["video_ids"]
        assert csv_eng.search("0.98")["video_ids"] == []  # confidence column ignored
        assert csv_eng.search("xe")["video_ids"] == []

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
