from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def make_plan_cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PlanCache:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, key: str, plan: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
