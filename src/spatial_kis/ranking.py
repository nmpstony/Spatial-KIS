from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import Iterable


@dataclass(frozen=True)
class Hit:
    keyframe_id: str
    score: float
    metadata: dict


def normalized_ip(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def condition_best(hits_by_region: Iterable[list[Hit]]) -> dict[str, Hit]:
    """Spatial groups are OR: a keyframe receives its best matching selected region."""
    best: dict[str, Hit] = {}
    for hits in hits_by_region:
        for hit in hits:
            prior = best.get(hit.keyframe_id)
            if prior is None or hit.score > prior.score:
                best[hit.keyframe_id] = hit
    return best


def weighted_harmonic(values: list[float], weights: list[float]) -> float:
    if len(values) != len(weights) or not values or any(value <= 0 for value in values):
        return 0.0
    return sum(weights) / sum(weight / value for value, weight in zip(values, weights))


def conjunction(condition_hits: list[dict[str, Hit]], weights: list[float]) -> list[dict]:
    """AND across conditions; only keyframes present in every condition survive."""
    if not condition_hits:
        return []
    shared = set(condition_hits[0])
    for hits in condition_hits[1:]:
        shared.intersection_update(hits)
    results = []
    for keyframe_id in shared:
        items = [hits[keyframe_id] for hits in condition_hits]
        score = weighted_harmonic([normalized_ip(item.score) for item in items], weights)
        result = {**items[0].metadata, "keyframe_id": keyframe_id, "score": score,
                  "condition_scores": [item.score for item in items]}
        results.append(result)
    return sorted(results, key=lambda item: item["score"], reverse=True)


def diversify_by_shot(results: list[dict], limit: int) -> list[dict]:
    """Round-robin shots, with a deterministic escape hatch for a ranked raw list in the API."""
    queues: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for result in results:
        queues[(result["video_id"], result["shot_id"])].append(result)
    output: list[dict] = []
    while queues and len(output) < limit:
        for key in list(queues):
            output.append(queues[key].pop(0))
            if not queues[key]:
                del queues[key]
            if len(output) == limit:
                break
    return output
