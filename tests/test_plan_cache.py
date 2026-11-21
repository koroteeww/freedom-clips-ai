from pathlib import Path

from freedom_clips.plan_cache import PlanCache, make_plan_cache_key


def test_cache_key_is_deterministic_and_cache_roundtrips(tmp_path: Path):
    a = {"model": "x", "settings": {"b": 2, "a": 1}}
    b = {"settings": {"a": 1, "b": 2}, "model": "x"}
    assert make_plan_cache_key(a) == make_plan_cache_key(b)
    cache = PlanCache(tmp_path)
    key = make_plan_cache_key(a)
    cache.save(key, {"clips": [1]})
    assert cache.load(key) == {"clips": [1]}
