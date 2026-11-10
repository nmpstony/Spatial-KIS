from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from .expander import variants
from .local_index import LocalMemmapIndex, clean_video_name
from .ranking import Hit, condition_best, conjunction, diversify_by_shot
from .regions import region_ids

ARTIFACT = Path(os.environ.get("SPATIAL_KIS_ARTIFACT", "artifacts/current"))


def find_video_root() -> Path:
    env_root = os.environ.get("SPATIAL_KIS_VIDEO_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    for candidate in [
        Path("videos"),
        Path("../videos"),
        Path("/home/team10/dataset"),
        Path(__file__).resolve().parents[2] / "videos",
        Path(__file__).resolve().parents[3] / "videos",
        Path("D:/Courses/HK7/CS336/videos"),
    ]:
        if candidate.is_dir():
            return candidate.resolve()
    return Path("videos").resolve()


VIDEO_ROOT = find_video_root()
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
BACKEND_MODE = os.environ.get("SPATIAL_KIS_BACKEND", "auto").lower()
HISTORY_PATH = ARTIFACT / "session_history.jsonl"


class Condition(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    region: str = "all"
    weight: float = Field(default=1.0, gt=0, le=10)
    stage: int = Field(default=1, ge=1, le=5)
    variants: list[str] | None = None


class SearchRequest(BaseModel):
    conditions: list[Condition] = Field(min_length=1, max_length=6)
    top_k: int = Field(default=300, ge=20, le=2048)
    result_limit: int = Field(default=60, ge=1, le=200)
    diversify: bool = True
    expand: bool = True
    max_temporal_gap: float = Field(default=45.0, ge=1.0, le=300.0)


class Feedback(BaseModel):
    keyframe_id: str
    label: str = Field(pattern="^(correct|near|irrelevant)$")


class DresLoginRequest(BaseModel):
    base_url: str = "https://eventretrieval.one"
    username: str
    password: str


class DresSubmitRequest(BaseModel):
    base_url: str = "https://eventretrieval.one"
    session_id: str
    evaluation_id: str
    mode: str = Field(default="kis", pattern="^(kis|qa)$")
    video_id: str
    start_ms: int = Field(ge=0)
    end_ms: int | None = None
    qa_answer: str | None = None


class Encoder:
    def __init__(self) -> None:
        import open_clip
        import threading

        self._lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

    @torch.inference_mode()
    def text(self, text: str) -> list[float]:
        with self._lock:
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
        if (ARTIFACT / "manifest.jsonl").exists():
            return {"active_collection": "local_memmap", "search_params": {}}
        raise RuntimeError("No active_index.json or manifest.jsonl found in artifact.") from error


@lru_cache
def client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


@lru_cache
def local_engine() -> LocalMemmapIndex:
    return LocalMemmapIndex(ARTIFACT)


def use_local_engine() -> bool:
    if BACKEND_MODE == "local":
        return True
    if BACKEND_MODE == "milvus":
        return False
    if (ARTIFACT / "memmap_index" / "meta.json").exists():
        return True
    try:
        client().list_collections()
        return False
    except Exception:
        return True


def resolve_condition_texts(condition: Condition, expand: bool) -> list[str]:
    if condition.variants:
        cleaned = [v.strip() for v in condition.variants if v and v.strip()]
        if cleaned and not (expand and len(cleaned) == 1 and cleaned[0] == condition.text.strip()):
            return cleaned[:3]
    if expand:
        expanded = variants(condition.text)
        # Prioritize English translation for OpenCLIP while keeping original if distinct
        if len(expanded) > 1:
            return list(dict.fromkeys([expanded[-1], *expanded]))[:3]
        return expanded[:3]
    return [condition.text.strip()]


def to_hit(row: dict) -> Hit:
    entity = row.get("entity", row)
    ts = float(entity["timestamp"])
    vid = entity["video_id"]
    return Hit(
        keyframe_id=entity["keyframe_id"],
        score=float(row.get("distance", row.get("score", 0))),
        metadata={
            "video_id": vid,
            "video_stem": clean_video_name(vid),
            "shot_id": entity["shot_id"],
            "frame_id": entity["frame_id"],
            "timestamp": ts,
            "timestamp_ms": int(round(ts * 1000)),
            "image_path": entity["image_path"],
            "region_id": entity["region_id"],
        },
    )


def run_conditions(conditions: list[Condition], top_k: int, expand: bool) -> list[dict[str, Hit]]:
    collection = state()["active_collection"]
    service = client()
    output: list[dict[str, Hit]] = []
    for condition in conditions:
        by_region: list[list[Hit]] = []
        text_variants = resolve_condition_texts(condition, expand)
        query_vecs = [encoder().text(text) for text in text_variants]
        for region in region_ids(condition.region):
            region_hits: list[Hit] = []
            for q_vec in query_vecs:
                rows = service.search(
                    collection_name=collection,
                    data=[q_vec],
                    anns_field="embedding",
                    limit=top_k,
                    filter=f"region_id == {region}",
                    output_fields=["keyframe_id", "video_id", "shot_id", "frame_id", "timestamp", "image_path", "region_id"],
                    search_params={"metric_type": "IP", "params": state().get("search_params", {})},
                )[0]
                region_hits.extend(to_hit(row) for row in rows)
            by_region.append(region_hits)
        output.append(condition_best(by_region))
    return output


def search(request: SearchRequest) -> dict:
    if use_local_engine():
        engine = local_engine()
        evaluated: list[tuple[tuple[int, ...], list[list[float]], float, int]] = []
        for cond in request.conditions:
            texts = resolve_condition_texts(cond, request.expand)
            q_vecs = [encoder().text(t) for t in texts]
            evaluated.append((region_ids(cond.region), q_vecs, cond.weight, cond.stage))
        return engine.search_conditions(
            evaluated_conditions=evaluated,
            top_k=request.top_k,
            result_limit=request.result_limit,
            diversify=request.diversify,
            max_temporal_gap=request.max_temporal_gap,
        )

    first = conjunction(
        run_conditions(request.conditions, request.top_k, request.expand),
        [item.weight for item in request.conditions],
    )
    results = diversify_by_shot(first, request.result_limit) if request.diversify else first[:request.result_limit]
    return {"kind": "frames", "results": results, "raw_count": len(first)}


app = FastAPI(title="Spatial KIS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    backend = "local_memmap" if use_local_engine() else "milvus"
    return {
        "ok": True,
        "backend": backend,
        "artifact": str(ARTIFACT),
        "collection": state().get("active_collection", backend),
    }


@app.post("/api/search")
def search_endpoint(request: SearchRequest) -> dict:
    result = search(request)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"request": request.model_dump(), "result_count": len(result["results"])}, ensure_ascii=False) + "\n")
    return result


@app.get("/api/neighbors")
def neighbors_endpoint(
    video_id: str = Query(...),
    timestamp: float = Query(default=0.0),
    window: float = Query(default=30.0, ge=2.0, le=300.0),
) -> dict:
    engine = local_engine()
    items = engine.video_neighbors(video_id=video_id, timestamp=timestamp, window_seconds=window, limit=30)
    return {"video_id": video_id, "video_stem": clean_video_name(video_id), "neighbors": items}


@app.post("/api/expand")
def expand_endpoint(condition: Condition) -> dict:
    return {"original": condition.text, "variants": variants(condition.text)}


@app.post("/api/feedback")
def feedback(request: Feedback) -> dict:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"feedback": request.model_dump()}, ensure_ascii=False) + "\n")
    return {"ok": True}


def _dres_json_request(url: str, method: str = "GET", payload: dict | None = None) -> dict | list:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=err.code, detail=f"DRES HTTP {err.code}: {detail}") from err
    except Exception as err:
        raise HTTPException(status_code=502, detail=f"DRES connection error: {err}") from err


@app.post("/api/dres/login")
def dres_login(req: DresLoginRequest) -> dict:
    base = req.base_url.rstrip("/")
    login_data = _dres_json_request(
        f"{base}/api/v2/login",
        method="POST",
        payload={"username": req.username, "password": req.password},
    )
    assert isinstance(login_data, dict)
    session_id = login_data.get("sessionId")
    evaluations: list = []
    if session_id:
        try:
            eval_res = _dres_json_request(
                f"{base}/api/v2/client/evaluation/list?session={urllib.parse.quote(str(session_id))}",
                method="GET",
            )
            if isinstance(eval_res, list):
                evaluations = eval_res
        except Exception:
            pass
    return {"login": login_data, "sessionId": session_id, "evaluations": evaluations}


@app.get("/api/dres/evaluations")
def dres_evaluations(base_url: str = Query("https://eventretrieval.one"), session_id: str = Query(...)) -> dict:
    base = base_url.rstrip("/")
    eval_res = _dres_json_request(
        f"{base}/api/v2/client/evaluation/list?session={urllib.parse.quote(session_id)}",
        method="GET",
    )
    return {"evaluations": eval_res if isinstance(eval_res, list) else []}


@app.post("/api/dres/submit")
def dres_submit(req: DresSubmitRequest) -> dict:
    base = req.base_url.rstrip("/")
    video_stem = clean_video_name(req.video_id)
    start_ms = int(req.start_ms)
    end_ms = int(req.end_ms if req.end_ms is not None else start_ms)

    if req.mode == "qa":
        answer_text = (req.qa_answer or "").strip()
        if not answer_text:
            raise HTTPException(400, "QA mode requires qa_answer")
        formatted_qa = f"{answer_text}-{video_stem}-{start_ms}"
        body = {"answerSets": [{"answers": [{"text": formatted_qa}]}]}
    else:
        body = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": video_stem,
                            "start": start_ms,
                            "end": end_ms,
                        }
                    ]
                }
            ]
        }

    url = f"{base}/api/v2/submit/{urllib.parse.quote(req.evaluation_id)}?session={urllib.parse.quote(req.session_id)}"
    response = _dres_json_request(url, method="POST", payload=body)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"dres_submit": req.model_dump(), "payload": body, "response": response}, ensure_ascii=False)
            + "\n"
        )
    return {"ok": True, "payload": body, "dres_response": response}


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

    stem = clean_video_name(video_id)
    candidates = [
        VIDEO_ROOT / video_id,
        VIDEO_ROOT / f"{video_id}.mp4",
        VIDEO_ROOT / f"{stem}.mp4",
        VIDEO_ROOT / stem.split("_")[0] / f"{stem}.mp4",
        VIDEO_ROOT / "videos" / f"{stem}.mp4",
    ]
    root = VIDEO_ROOT.resolve()
    for candidate in candidates:
        if candidate.is_file():
            target = candidate.resolve()
            if root in target.parents or target.parent == root:
                return FileResponse(target, media_type="video/mp4")
    # Recursive fallback inside VIDEO_ROOT if nested in L01..L30 subfolder
    matches = list(root.rglob(f"{stem}.mp4"))
    if matches:
        return FileResponse(matches[0].resolve(), media_type="video/mp4")
    raise HTTPException(404, f"Video not found: {video_id}")


WEB = Path(__file__).parents[2] / "web"
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
