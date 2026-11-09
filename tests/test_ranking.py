from spatial_kis.ranking import Hit, condition_best, conjunction


def hit(key, score, **metadata):
    return Hit(key, score, {"video_id": "v.mp4", "shot_id": 1, "timestamp": 10.0, **metadata})


def test_region_group_uses_best_or_hit():
    output = condition_best([[hit("a", 0.2)], [hit("a", 0.8), hit("b", 0.4)]])
    assert output["a"].score == 0.8


def test_conjunction_requires_every_condition():
    result = conjunction([{"a": hit("a", 0.8), "b": hit("b", 0.8)}, {"a": hit("a", 0.4)}], [1, 1])
    assert [item["keyframe_id"] for item in result] == ["a"]
