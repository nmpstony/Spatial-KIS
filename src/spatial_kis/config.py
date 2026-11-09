from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    model_name: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"
    candidate_fps: float = 2.0
    semantic_distance_threshold: float = 0.13
    max_gap_seconds: float = 8.0
    min_sharpness: float = 55.0
    grid_overlap: float = 0.12
    webp_quality: int = 85
    long_edge: int = 1280
    batch_size: int = 48
    device: str = "auto"

    @classmethod
    def load(cls, path: Path) -> "PreprocessConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def as_dict(self) -> dict:
        return asdict(self)


def load_profiles(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
