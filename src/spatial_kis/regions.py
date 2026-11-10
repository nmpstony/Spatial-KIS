from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image

REGION_NAMES = (
    "all",
    "top_left", "top_center", "top_right",
    "middle_left", "center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
)

REGION_GROUPS = {
    "all": (0,),
    "whole": (0,),
    "left": (1, 4, 7),
    "right": (3, 6, 9),
    "top": (1, 2, 3),
    "bottom": (7, 8, 9),
    "middle_row": (4, 5, 6),
    "center_col": (2, 5, 8),
    "center": (5,),
    "top_left": (1,),
    "top_center": (2,),
    "top_right": (3,),
    "middle_left": (4,),
    "middle_right": (6,),
    "bottom_left": (7,),
    "bottom_center": (8,),
    "bottom_right": (9,),
}


def region_ids(selection: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(selection, str):
        if selection not in REGION_GROUPS:
            raise ValueError(f"Unknown spatial selection: {selection}")
        return REGION_GROUPS[selection]
    ids = tuple(selection)
    if not ids or any(region < 0 or region > 9 for region in ids):
        raise ValueError("Region IDs must be in 0..9")
    return ids


def grid_crops(image: Image.Image, overlap: float) -> list[Image.Image]:
    """Return row-major 3x3 crops, with edge-clamped overlap."""
    if not 0 <= overlap < 0.5:
        raise ValueError("overlap must be in [0, 0.5)")
    width, height = image.size
    xs = (0, width // 3, (2 * width) // 3, width)
    ys = (0, height // 3, (2 * height) // 3, height)
    x_pad, y_pad = int(width * overlap / 3), int(height * overlap / 3)
    crops: list[Image.Image] = []
    for row in range(3):
        for column in range(3):
            left = max(0, xs[column] - x_pad)
            right = min(width, xs[column + 1] + x_pad)
            top = max(0, ys[row] - y_pad)
            bottom = min(height, ys[row + 1] + y_pad)
            crops.append(image.crop((left, top, right, bottom)))
    return crops


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)
