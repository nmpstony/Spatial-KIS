"""
Extract Vietnamese & English OCR text from Spatial-KIS keyframes into ocr.csv.

Usage (on Kaggle GPU or local/server):
    pip install easyocr
    python scripts/build_ocr_csv.py --artifact /path/to/spatial-kis-artifact --output ocr.csv --stride 3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ocr.csv from Spatial-KIS artifact keyframes")
    parser.add_argument("--artifact", required=True, type=Path, help="Path to spatial-kis-artifact directory")
    parser.add_argument("--output", default="ocr.csv", type=Path, help="Output CSV path")
    parser.add_argument("--stride", default=3, type=int, help="Sample 1 every N keyframes per video (default: 3)")
    parser.add_argument("--batch-prefix", default="", type=str, help="Optional video prefix filter, e.g. L21,L22,L23,L24")
    args = parser.parse_args()

    import easyocr
    import numpy as np

    reader = easyocr.Reader(["vi", "en"], gpu=True)
    manifest_path = args.artifact / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.jsonl not found in {args.artifact}")

    prefixes = [p.strip().upper() for p in args.batch_prefix.split(",") if p.strip()]
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = rec["video_id"].replace(".mp4", "").upper()
            if prefixes and not any(vid.startswith(p) for p in prefixes):
                continue
            records.append(rec)

    sampled = records[:: max(1, args.stride)]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["video_id", "frame_id", "timestamp", "text"])
        last_text_by_video: dict[str, str] = {}

        for rec in tqdm(sampled, desc="OCR Keyframes"):
            img_path = args.artifact / rec["image_path"]
            if not img_path.is_file():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.asarray(img)
                texts = reader.readtext(arr, detail=0, paragraph=True)
                joined = " | ".join(t.strip() for t in texts if t and len(t.strip()) >= 2)
                if not joined:
                    continue
                vid = rec["video_id"].replace(".mp4", "")
                if last_text_by_video.get(vid) == joined:
                    continue
                last_text_by_video[vid] = joined
                writer.writerow([vid, rec["frame_id"], round(float(rec["timestamp"]), 2), joined])
            except Exception:
                continue


if __name__ == "__main__":
    main()
