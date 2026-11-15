import json
import gc
import tempfile
from pathlib import Path
import numpy as np
from spatial_kis.local_index import (
    CsvSearchEngine,
    LocalMemmapIndex,
    _asr_aligned_segments,
    clean_video_name,
)


def test_clean_video_name_strips_extension_and_directory():
    assert clean_video_name("L01_V001.mp4") == "L01_V001"
    assert clean_video_name("L01/L01_V002.MP4") == "L01_V002"
    assert clean_video_name("L02_V010") == "L02_V010"


def test_local_memmap_index_builds_and_searches():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
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
        del csv_eng
        idx._region_tensor_fp32.cache_clear()
        for memmap in idx._mmaps.values():
            memmap._mmap.close()
        idx._mmaps.clear()
        del idx
        gc.collect()


def test_csv_engines_isolate_sources_and_parse_asr_intervals():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "csv").mkdir()
        (root / "OCR_main.CSV").write_text(
            "video_id,frame_id,timestamp,text\nL01_V001,10,1.0,Tuyên Quang OCR\n",
            encoding="utf-8",
        )
        (root / "csv" / "asr_segments.csv").write_text(
            "video_id,start,end,text\n"
            "L02_V001,2.500,5.500,Thành phố Tuyên Quang năm 2024 chặng 13\n"
            "L02_V002,8.000,9.000,thanh pho tuyen quan\n",
            encoding="utf-8",
        )
        (root / "asr_ms.csv").write_text(
            "video_id,timestamp_ms,end_ms,text\nL03_V001,1250,2750,Bản tin buổi sáng\n",
            encoding="utf-8",
        )

        ocr = CsvSearchEngine(root)
        asr = CsvSearchEngine(root, source="asr")

        assert ocr.status()["source"] == "ocr"
        assert [item["name"] for item in ocr.status()["files"]] == ["OCR_main.CSV"]
        assert asr.status()["source"] == "asr"
        assert {item["name"] for item in asr.status()["files"]} == {"asr_segments.csv", "asr_ms.csv"}
        assert ocr.search("ban tin")["video_ids"] == []
        assert asr.search("Tuyên Quang")["video_ids"] == ["L02_V001", "L02_V002"]

        fuzzy = asr.search("tuyen quang")
        assert "L02_V002" in fuzzy["video_ids"]  # one missing suffix consonant in ASR
        assert fuzzy["source"] == "asr"
        assert all(match["source"] == "asr" for match in fuzzy["matches"])

        # Long ASR phrases use the row-signature index instead of bloating span postings.
        assert "thanhphotuyenquan" not in asr.vocab_postings
        assert "L02_V002" in asr.search("thanh pho tuyen quang")["video_ids"]

        exact_digits = asr.search("nam 2024")
        assert exact_digits["video_ids"] == ["L02_V001"]
        assert asr.search("nam 2o24")["video_ids"] == []
        assert asr.search("chang i3")["video_ids"] == []

        seconds_match = asr.search("Tuyên Quang")["matches"][0]
        assert seconds_match["timestamp"] == 2.5
        assert seconds_match["end_timestamp"] == 5.5
        assert seconds_match["timestamp_ms"] == 2500
        assert seconds_match["end_ms"] == 5500
        assert seconds_match["frame_id"] is None
        assert seconds_match["image_path"] is None

        ms_match = asr.search("ban tin buoi sang")["matches"][0]
        assert ms_match["timestamp"] == 1.25
        assert ms_match["end_timestamp"] == 2.75
        assert ms_match["timestamp_ms"] == 1250
        assert ms_match["end_ms"] == 2750

        # Ordered-token match skips natural filler words in spoken ASR transcripts while staying strict on digits
        assert ms_match["video_id"] in asr.search("bản tin sáng")["video_ids"]
        assert asr.search("2024 Tuyên Quang")["video_ids"] == []

        del ocr
        del asr
        gc.collect()


def test_asr_phrase_gap_diacritics_and_short_name_precision():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "asr_precision.csv").write_text(
            "video_id,start,end,text\n"
            "L01_V001,0,1,Tây có rất nhiều vùng đất ở Tây Nguyên\n"
            "L01_V002,1,2,Tây hôm nay có nhiều tin tức từ vùng cao nguyên\n"
            "L01_V003,2,3,Thông tin mới nhất từ Điện Biên\n"
            "L01_V004,3,4,Dự báo về diễn biến thời tiết\n"
            "L01_V005,4,5,News from dien bien today\n"
            "L01_V006,5,6,thanh pho tuyen quan\n"
            "L01_V007,6,7,Du lịch Sầm Sơn mùa hè\n"
            "L01_V008,7,8,Người dân cầm sơn sửa nhà\n"
            "L01_V009,8,9,Chúng ta làm sao bây giờ\n"
            "L01_V010,9,10,Tầng số 1 đang mở cửa\n",
            encoding="utf-8",
        )
        asr = CsvSearchEngine(root, source="asr")

        tay_nguyen = asr.search("tây nguyên")["video_ids"]
        assert "L01_V001" in tay_nguyen
        assert "L01_V002" not in tay_nguyen

        dien_bien = asr.search("điện biên")["video_ids"]
        assert "L01_V003" in dien_bien
        assert "L01_V005" in dien_bien
        assert "L01_V004" not in dien_bien
        assert "L01_V006" in asr.search("Tuyên Quang")["video_ids"]

        sam_son = asr.search("sầm sơn")["video_ids"]
        assert "L01_V007" in sam_son
        assert "L01_V008" not in sam_son
        assert "L01_V009" not in sam_son

        assert asr.search("số 1")["video_ids"] == ["L01_V010"]

        del asr
        gc.collect()


def test_asr_tokenizer_keeps_normalized_and_raw_tokens_aligned():
    text = "Nhà thờ Sacré-Cœur diện tích 100 m² tại CO₂ ở Hà Nội"
    text_norm, segments, raw_segments = _asr_aligned_segments(text)

    assert text_norm == "nha tho sacre coeur dien tich 100 m tai co o ha noi"
    assert all(
        len(segment) == len(raw_segment)
        for segment, raw_segment in zip(segments, raw_segments)
    )
    assert len(segments) == len(raw_segments)
    assert "coeur" in segments[0]
    assert "cœur" in raw_segments[0]


def test_api_csv_and_asr_helpers_preserve_intervals_and_upload_prefixes(monkeypatch):
    try:
        from spatial_kis import api
    except (ImportError, RuntimeError) as exc:
        import pytest

        pytest.skip(f"FastAPI backend dependencies unavailable: {exc}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "ocr.csv").write_text(
            "video_id,frame_id,timestamp,text\nL01_V001,10,4.0,Tuyên Quang OCR\n",
            encoding="utf-8",
        )
        (root / "asr.csv").write_text(
            "video_id,start,end,text\nL01_V001,10.0,14.0,Tuyên Quang ASR\n",
            encoding="utf-8",
        )

        class FakeLocalIndex:
            def resolve_allowed_videos(self, filters):
                return {value.upper() for value in filters}

            def video_neighbors(self, video_id, timestamp, window_seconds, limit):
                assert video_id == "L01_V001"
                assert timestamp == 12.0
                assert window_seconds == 15.0
                assert limit == 5
                return [
                    {"timestamp": 9.0, "frame_id": 90, "image_path": "images/L01_V001/frame_90.webp"},
                    {"timestamp": 12.5, "frame_id": 125, "image_path": "images/L01_V001/frame_125.webp"},
                ]

        monkeypatch.setattr(api, "ARTIFACT", root)
        monkeypatch.setattr(api, "use_local_engine", lambda: True)
        monkeypatch.setattr(api, "local_engine", lambda: FakeLocalIndex())
        api.csv_engine.cache_clear()
        api.asr_engine.cache_clear()
        try:
            assert api.csv_status_endpoint()["source"] == "ocr"
            assert api.asr_status_endpoint()["source"] == "asr"
            csv_result = api.csv_search_endpoint(api.CsvSearchRequest(query="tuyen quang"))
            assert csv_result["video_ids"] == ["L01_V001"]

            asr_result = api.asr_search_endpoint(api.CsvSearchRequest(query="tuyen quang"))
            match = asr_result["matches"][0]
            assert match["timestamp"] == 10.0
            assert match["end_timestamp"] == 14.0
            assert match["timestamp_ms"] == 10000
            assert match["end_ms"] == 14000
            assert match["keyframe_timestamp"] == 12.5
            assert match["frame_id"] == 125
            assert match["image_path"].endswith("frame_125.webp")

            asr_upload = api.asr_upload_endpoint(
                api.CsvUploadRequest(filename="segments.csv", content="video_id,start,end,text\nL02_V001,1,2,hello\n")
            )
            ocr_upload = api.csv_upload_endpoint(
                api.CsvUploadRequest(filename="notes.csv", content="video_id,timestamp,text\nL03_V001,1,hello\n")
            )
            assert asr_upload["saved_as"] == "asr_segments.csv"
            assert ocr_upload["saved_as"] == "ocr_notes.csv"
            assert (root / "asr_segments.csv").is_file()
            assert (root / "ocr_notes.csv").is_file()
            assert {item["name"] for item in api.asr_engine().status()["files"]} == {"asr.csv", "asr_segments.csv"}
            assert {item["name"] for item in api.csv_engine().status()["files"]} == {"ocr.csv", "ocr_notes.csv"}
        finally:
            api.csv_engine.cache_clear()
            api.asr_engine.cache_clear()
            gc.collect()
