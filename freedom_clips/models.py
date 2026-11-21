from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True, frozen=True)
class Unit:
    unit_id: int
    start: float
    end: float
    duration: float
    text: str
