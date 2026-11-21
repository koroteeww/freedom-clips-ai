from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .models import Cue, Unit

ALLOWED_EFFECTS = {
    "none",
    "punch_in",
    "contrast_boost",
    "crop_left",
    "crop_right",
    "flash_cut",
}

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class PlanValidationError(ValueError):
    pass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sanitize_filename(text: str, max_len: int = 90) -> str:
    text = str(text or "").strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip(" ._")[:max_len].rstrip(" .")
    if not text:
        text = "clip"
    if text.upper() in _WINDOWS_RESERVED:
        text = f"_{text}"
    return text


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise PlanValidationError(f"{name} must be finite")
    return number


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _decimal_minutes_to_seconds(value: float) -> float | None:
    """Interpret an ambiguous MM.SS-style number as seconds, if possible."""
    if value < 1 or value >= 60 or float(value).is_integer():
        return None
    minutes = int(value)
    seconds_text = f"{value:.6f}".split(".", 1)[1].rstrip("0")
    if not seconds_text:
        return None
    seconds = int(seconds_text[:2].ljust(2, "0"))
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


def _looks_like_decimal_minute_range(
    raw_start: float,
    raw_end: float,
    video_duration: float,
    min_sec: float,
    max_sec: float,
) -> bool:
    """Detect common LLM mistake: 26.54 means 00:26:54, not 26.54 seconds."""
    if raw_end - raw_start >= min_sec:
        return False
    converted_start = _decimal_minutes_to_seconds(raw_start)
    converted_end = _decimal_minutes_to_seconds(raw_end)
    if converted_start is None or converted_end is None or converted_end <= converted_start:
        return False
    converted_duration = converted_end - converted_start
    return min_sec <= converted_duration <= max_sec and converted_end <= video_duration + 1e-6


def _fit_candidate_to_cues(
    raw_start: float,
    raw_end: float,
    cues: list[Cue],
    video_duration: float,
    min_sec: float,
    max_sec: float,
) -> tuple[float, float] | None:
    if not cues:
        return None

    start_candidates = sorted(cues, key=lambda c: abs(c.start - raw_start))[:8]
    best: tuple[float, float, float] | None = None
    for start_cue in start_candidates:
        start = max(0.0, start_cue.start)
        for end_cue in cues:
            end = min(video_duration, end_cue.end)
            duration = end - start
            if duration < min_sec or duration > max_sec:
                continue
            penalty = abs(start - raw_start) + abs(end - raw_end)
            if best is None or penalty < best[0]:
                best = (penalty, start, end)
    return None if best is None else (best[1], best[2])


def validate_contiguous_clips(
    raw_clips: list[dict[str, Any]],
    cues: list[Cue],
    video_duration: float,
    min_sec: float,
    max_sec: float,
    max_clips: int,
    max_overlap_sec: float = 0.0,
) -> list[dict[str, Any]]:
    if min_sec <= 0 or max_sec < min_sec:
        raise ValueError("Invalid duration bounds")

    normalized: list[dict[str, Any]] = []
    for raw in raw_clips:
        try:
            raw_start = _finite_number(raw.get("start_sec"), "start_sec")
            raw_end = _finite_number(raw.get("end_sec"), "end_sec")
        except PlanValidationError:
            continue
        if raw_end <= raw_start:
            continue
        if _looks_like_decimal_minute_range(raw_start, raw_end, video_duration, min_sec, max_sec):
            continue

        fitted = _fit_candidate_to_cues(
            raw_start, raw_end, cues, video_duration, min_sec, max_sec
        )
        if fitted is None:
            continue
        start, end = fitted

        score = int(clamp(_finite_number(raw.get("viral_score", 0), "viral_score"), 0, 100))
        title = str(raw.get("title") or "Untitled clip").strip()[:140]
        hook = str(raw.get("hook_text") or title).strip()[:160]
        caption = str(raw.get("caption") or "").strip()[:500]
        why = str(raw.get("why_selected") or "").strip()[:500]

        normalized.append(
            {
                "rank": int(raw.get("rank") or len(normalized) + 1),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "viral_score": score,
                "title": title,
                "hook_text": hook,
                "caption": caption,
                "why_selected": why,
            }
        )

    normalized.sort(key=lambda item: item["viral_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in normalized:
        if len(selected) >= max_clips:
            break
        conflict = False
        for chosen in selected:
            overlap = overlap_seconds(
                candidate["start_sec"], candidate["end_sec"],
                chosen["start_sec"], chosen["end_sec"],
            )
            if overlap > max_overlap_sec + 1e-6:
                conflict = True
                break
        if not conflict:
            selected.append(candidate)

    for rank, clip in enumerate(selected, start=1):
        clip["rank"] = rank
    return selected


def validate_montage_plan(
    raw_plan: dict[str, Any],
    units_by_id: dict[int, Unit],
    min_clip_sec: float,
    max_clip_sec: float,
    edit_mode: str,
    max_clips: int,
) -> dict[str, Any]:
    if edit_mode not in {"chronological", "montage"}:
        raise ValueError("edit_mode must be chronological or montage")
    clips = raw_plan.get("clips")
    if not isinstance(clips, list):
        raise PlanValidationError("Plan must contain a clips array")

    clean_clips: list[dict[str, Any]] = []
    for index, raw_clip in enumerate(clips, start=1):
        raw_beats = raw_clip.get("beats")
        if not isinstance(raw_beats, list) or not (3 <= len(raw_beats) <= 6):
            continue

        beats: list[dict[str, Any]] = []
        unit_ids: list[int] = []
        valid = True
        for raw_beat in raw_beats:
            try:
                unit_id = int(raw_beat["unit_id"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if unit_id not in units_by_id:
                valid = False
                break
            effect = str(raw_beat.get("effect") or "none")
            if effect not in ALLOWED_EFFECTS:
                effect = "none"
            speed = clamp(_finite_number(raw_beat.get("speed", 1.0), "speed"), 1.0, 1.18)
            beats.append(
                {
                    "unit_id": unit_id,
                    "effect": effect,
                    "speed": round(speed, 4),
                    "overlay": str(raw_beat.get("overlay") or "").strip()[:80],
                    "reason": str(raw_beat.get("reason") or "").strip()[:300],
                }
            )
            unit_ids.append(unit_id)

        if not valid:
            continue
        if edit_mode == "chronological" and unit_ids != sorted(unit_ids):
            continue

        duration = sum(units_by_id[b["unit_id"]].duration / b["speed"] for b in beats)
        if not (min_clip_sec <= duration <= max_clip_sec):
            continue

        hook = raw_clip.get("hook") or {}
        hook_text = str(hook.get("text") or raw_clip.get("title") or "Watch this").strip()[:140]
        display_seconds = clamp(
            _finite_number(hook.get("display_seconds", 2.2), "hook.display_seconds"),
            1.0,
            3.0,
        )
        score = int(clamp(_finite_number(raw_clip.get("viral_score", 0), "viral_score"), 0, 100))
        integrity = int(
            clamp(_finite_number(raw_clip.get("integrity_risk", 0), "integrity_risk"), 0, 100)
        )

        clean_clips.append(
            {
                "clip_id": int(raw_clip.get("clip_id") or index),
                "title": str(raw_clip.get("title") or f"Clip {index}").strip()[:140],
                "why_selected": str(raw_clip.get("why_selected") or "").strip()[:500],
                "viral_score": score,
                "integrity_risk": integrity,
                "hook": {
                    "display_seconds": round(display_seconds, 3),
                    "text": hook_text,
                },
                "beats": beats,
                "duration_sec": round(duration, 3),
                "cta_question": str(raw_clip.get("cta_question") or "").strip()[:160],
            }
        )

    clean_clips.sort(key=lambda item: item["viral_score"], reverse=True)
    clean_clips = clean_clips[:max_clips]
    if not clean_clips:
        raise PlanValidationError("No valid montage clips remained after validation")
    return {
        "global_theme": str(raw_plan.get("global_theme") or "").strip()[:300],
        "clips": clean_clips,
    }


def require_existing_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
