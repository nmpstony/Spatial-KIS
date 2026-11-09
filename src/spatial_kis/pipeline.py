from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from scenedetect import AdaptiveDetector, detect
from tqdm import tqdm

from .artifact import ArtifactWriter, KeyframeRecord
from .config import PreprocessConfig
from .regions import grid_crops, normalize


@dataclass
class Candidate:
    frame_id: int
    timestamp: float
    image: Image.Image
    sharpness: float
    shot_id: int


def orientation_degrees(video_path: Path) -> int:
    """Honor display rotation if ffprobe is available; zero is safe fallback."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream_side_data=rotation", "-of", "default=nw=1:nk=1", str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        return int(float(result.stdout.strip() or "0")) % 360
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return 0


def detect_shots(video_path: Path, duration: float) -> list[tuple[float, float]]:
    try:
        scenes = detect(str(video_path), AdaptiveDetector())
        output = [(start.get_seconds(), end.get_seconds()) for start, end in scenes]
        return output or [(0.0, duration)]
    except Exception:
        # Explicit fallback: candidate sampling still gives usable segments when shot detection fails.
        return [(0.0, duration)]


def sharpness(frame_bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def pil_from_frame(frame_bgr: np.ndarray, rotate: int) -> Image.Image:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return image.rotate(-rotate, expand=True) if rotate else image


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    if max(image.size) <= long_edge:
        return image
    scale = long_edge / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


class ClipEncoder:
    def __init__(self, config: PreprocessConfig):
        import open_clip

        device_name = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        self.device = torch.device(device_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(config.model_name, pretrained=config.pretrained)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        values: list[np.ndarray] = []
        for start in range(0, len(images), 96):
            tensors = torch.stack([self.preprocess(item) for item in images[start:start + 96]]).to(self.device)
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    encoded = self.model.encode_image(tensors)
            else:
                encoded = self.model.encode_image(tensors)
            values.append(encoded.float().cpu().numpy())
        return normalize(np.concatenate(values, axis=0))


def sampled_candidates(video_path: Path, shots: list[tuple[float, float]], fps: float, config: PreprocessConfig) -> list[Candidate]:
    cap = cv2.VideoCapture(str(video_path))
    step = max(1, round(fps / config.candidate_fps))
    rotation = orientation_degrees(video_path)
    candidates: list[Candidate] = []
    shot_index = 0
    frame_number = 0
    boundaries = [(round(start * fps), round(end * fps)) for start, end in shots]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while shot_index + 1 < len(boundaries) and frame_number >= boundaries[shot_index][1]:
            shot_index += 1
        start, end = boundaries[shot_index]
        boundary = frame_number in (start, max(start, end - 1))
        if boundary or (start <= frame_number < end and (frame_number - start) % step == 0):
            score = sharpness(frame)
            if score >= config.min_sharpness:
                candidates.append(Candidate(frame_number, frame_number / fps, pil_from_frame(frame, rotation), score, shot_index))
        frame_number += 1
    cap.release()
    return candidates


def retain_candidates(candidates: list[Candidate], encoder: ClipEncoder, config: PreprocessConfig) -> list[tuple[Candidate, np.ndarray]]:
    kept: list[tuple[Candidate, np.ndarray]] = []
    by_shot: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_shot.setdefault(candidate.shot_id, []).append(candidate)
    for shot in by_shot.values():
        features = encoder.encode_images([candidate.image for candidate in shot])
        latest_index = 0
        for index, (candidate, feature) in enumerate(zip(shot, features)):
            is_edge = index == 0 or index == len(shot) - 1
            distance = 1.0 - float(np.dot(features[latest_index], feature))
            max_gap = candidate.timestamp - shot[latest_index].timestamp >= config.max_gap_seconds
            if is_edge or distance >= config.semantic_distance_threshold or max_gap:
                kept.append((candidate, feature))
                latest_index = index
    return kept


def process_video(video_path: Path, videos_root: Path, writer: ArtifactWriter, encoder: ClipEncoder, config: PreprocessConfig) -> int:
    video_id = video_path.relative_to(videos_root).as_posix()
    done = writer.root / "checkpoints" / f"{video_id.replace('/', '__')}.json"
    if done.exists():
        return int(json.loads(done.read_text()).get("keyframes", 0))
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        raise ValueError("Video has no readable FPS")
    shots = detect_shots(video_path, frame_count / fps)
    retained = retain_candidates(sampled_candidates(video_path, shots, fps, config), encoder, config)
    vectors: list[np.ndarray] = []
    records: list[KeyframeRecord] = []
    image_dir = writer.root / "images" / video_id.replace(".mp4", "")
    image_dir.mkdir(parents=True, exist_ok=True)
    for offset, (candidate, global_vector) in enumerate(retained):
        display = resize_long_edge(candidate.image, config.long_edge)
        keyframe_id = f"{video_id}:{candidate.frame_id}"
        image_rel = f"images/{video_id.removesuffix('.mp4')}/frame_{candidate.frame_id}.webp"
        output_image = writer.root / image_rel
        output_image.parent.mkdir(parents=True, exist_ok=True)
        display.save(output_image, "WEBP", quality=config.webp_quality, method=6)
        local_vectors = encoder.encode_images(grid_crops(candidate.image, config.grid_overlap))
        vectors.append(np.concatenate((global_vector[None, :], local_vectors), axis=0))
        records.append(KeyframeRecord(keyframe_id, video_id, candidate.shot_id, candidate.frame_id, fps,
                                      candidate.timestamp, display.width, display.height, candidate.sharpness,
                                      image_rel, "", offset))
    if vectors:
        shard = writer.write_video_embeddings(video_id, np.stack(vectors))
        for record in records:
            record.embedding_shard = shard
            writer.append_manifest(record)
    done.parent.mkdir(exist_ok=True)
    done.write_text(json.dumps({"video_id": video_id, "keyframes": len(records)}), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a resumable Spatial KIS artifact")
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive index in sorted video list (for Kaggle chunks)")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive index in sorted video list")
    args = parser.parse_args()
    config = PreprocessConfig.load(args.config)
    all_video_paths = sorted(args.videos.glob(args.pattern))
    video_paths = all_video_paths[args.start_index:args.end_index]
    if not video_paths:
        raise SystemExit("No matching videos")
    writer = ArtifactWriter(args.artifact, uuid.uuid4().hex, config.as_dict())
    encoder = ClipEncoder(config)
    errors: list[dict] = []
    for path in tqdm(video_paths, desc="Videos"):
        try:
            process_video(path, args.videos, writer, encoder, config)
        except Exception as error:
            errors.append({"video_id": path.relative_to(args.videos).as_posix(), "error": str(error)})
    (args.artifact / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    writer.finalize(len(video_paths))
    print(json.dumps({"artifact": str(args.artifact), "selected_videos": len(video_paths), "total_videos": len(all_video_paths), "errors": len(errors)}))


if __name__ == "__main__":
    main()
