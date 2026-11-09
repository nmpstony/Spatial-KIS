from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from .ranking import Hit, condition_best, conjunction, diversify_by_shot
from .regions import region_ids
from .expander import variants

ARTIFACT = Path(os.environ.get("SPATIAL_KIS_ARTIFACT", "artifacts/current"))


def find_video_root() -> Path:
    env_root = os.environ.get("SPATIAL_KIS_VIDEO_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    for candidate in [
        Path("videos"),
        Path("../videos"),
        Path(__file__).resolve().parents[2] / "videos",
        Path(__file__).resolve().parents[3] / "videos",
        Path("D:/Courses/HK7/CS336/videos"),
    ]:
        if candidate.is_dir():
            return candidate.resolve()
    return Path("videos").resolve()


VIDEO_ROOT = find_video_root()
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
HISTORY_PATH = ARTIFACT / "session_history.jsonl"


class Condition(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    region: str = "all"
    weight: float = Field(default=1.0, gt=0, le=10)
    variants: list[str] | None = None


class SearchRequest(BaseModel):
    conditions: list[Condition] = Field(min_length=1, max_length=6)
    top_k: int = Field(default=300, ge=20, le=1024)
    result_limit: int = Field(default=60, ge=1, le=200)
    diversify: bool = True
    expand: bool = True


class Feedback(BaseModel):
    keyframe_id: str
    label: str = Field(pattern="^(correct|near|irrelevant)$")


class Encoder:
    def __init__(self) -> None:
        import open_clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

    @torch.inference_mode()
    def text(self, text: str) -> list[float]:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            value = self.model.encode_text(tokens)
        value = value / value.norm(dim=-1, keepdim=True)
        return value.float().cpu().numpy()[0].tolist()


@lru_cache
def encoder() -> Encoder:
    return Encoder()


@lru_cache
def state() -> dict:
    try:
        return json.loads((ARTIFACT / "active_index.json").read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError("No active_index.json. Import a verified artifact first.") from error


def client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


def to_hit(row: dict) -> Hit:
    entity = row.get("entity", row)
    return Hit(keyframe_id=entity["keyframe_id"], score=float(row.get("distance", row.get("score", 0))), metadata={
        "video_id": entity["video_id"], "shot_id": entity["shot_id"], "frame_id": entity["frame_id"],
        "timestamp": entity["timestamp"], "image_path": entity["image_path"], "region_id": entity["region_id"],
    })


def run_conditions(conditions: list[Condition], top_k: int, expand: bool) -> list[dict[str, Hit]]:
    collection = state()["active_collection"]
    service = client()
    output: list[dict[str, Hit]] = []
    for condition in conditions:
        by_region: list[list[Hit]] = []
        for region in region_ids(condition.region):
            region_hits: list[Hit] = []
            text_variants = condition.variants or (variants(condition.text) if expand else [condition.text])
            for text in text_variants[:3]:
                rows = service.search(collection_name=collection, data=[encoder().text(text)], anns_field="embedding",
                                      limit=top_k, filter=f"region_id == {region}",
                                      output_fields=["keyframe_id", "video_id", "shot_id", "frame_id", "timestamp", "image_path", "region_id"],
                                      search_params={"metric_type": "IP", "params": state().get("search_params", {})})[0]
                region_hits.extend(to_hit(row) for row in rows)
            by_region.append(region_hits)
        output.append(condition_best(by_region))
    return output


def search(request: SearchRequest) -> dict:
    first = conjunction(run_conditions(request.conditions, request.top_k, request.expand), [item.weight for item in request.conditions])
    results = diversify_by_shot(first, request.result_limit) if request.diversify else first[:request.result_limit]
    return {"kind": "frames", "results": results, "raw_count": len(first)}


app = FastAPI(title="Spatial KIS")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8010"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"ok": True, "artifact": str(ARTIFACT), "collection": state()["active_collection"]}


@app.post("/api/search")
def search_endpoint(request: SearchRequest) -> dict:
    result = search(request)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"request": request.model_dump(), "result_count": len(result["results"])}, ensure_ascii=False) + "\n")
    return result


@app.post("/api/expand")
def expand_endpoint(condition: Condition) -> dict:
    return {"original": condition.text, "variants": variants(condition.text)}


@app.post("/api/feedback")
def feedback(request: Feedback) -> dict:
    # Session-local UI consumes this acknowledgement; it intentionally never modifies the model or index.
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"feedback": request.model_dump()}, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/media/keyframe/{path:path}")
def keyframe_path(path: str):
    from fastapi.responses import FileResponse
    target = (ARTIFACT / "images" / path).resolve()
    root = (ARTIFACT / "images").resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


@app.get("/media/video/{video_id:path}")
def video_path(video_id: str):
    from fastapi.responses import FileResponse
    target = (VIDEO_ROOT / video_id).resolve()
    if not target.is_file() and not video_id.endswith(".mp4"):
        target = (VIDEO_ROOT / f"{video_id}.mp4").resolve()
    root = VIDEO_ROOT.resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404, f"Video not found: {video_id}")
    return FileResponse(target, media_type="video/mp4")


WEB = Path(__file__).parents[2] / "web"
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
