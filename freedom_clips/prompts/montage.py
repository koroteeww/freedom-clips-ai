from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..models import Unit
from ..validation import ALLOWED_EFFECTS

PROMPT_VERSION = "montage-original-footage-v2"


def get_montage_schema(n_clips: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "global_theme": {"type": "string"},
            "clips": {
                "type": "array",
                "minItems": n_clips,
                "maxItems": n_clips,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "clip_id": {"type": "integer"},
                        "title": {"type": "string"},
                        "why_selected": {"type": "string"},
                        "viral_score": {"type": "integer"},
                        "integrity_risk": {"type": "integer"},
                        "hook": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "display_seconds": {"type": "number"},
                                "text": {"type": "string"},
                            },
                            "required": ["display_seconds", "text"],
                        },
                        "beats": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 6,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "unit_id": {"type": "integer"},
                                    "effect": {"type": "string", "enum": sorted(ALLOWED_EFFECTS)},
                                    "speed": {"type": "number"},
                                    "overlay": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["unit_id", "effect", "speed", "overlay", "reason"],
                            },
                        },
                        "cta_question": {"type": "string"},
                    },
                    "required": [
                        "clip_id",
                        "title",
                        "why_selected",
                        "viral_score",
                        "integrity_risk",
                        "hook",
                        "beats",
                        "cta_question",
                    ],
                },
            },
        },
        "required": ["global_theme", "clips"],
    }


def build_instructions(
    *,
    n_clips: int,
    min_clip_sec: float,
    max_clip_sec: float,
    edit_mode: str,
    content_context: str,
) -> str:
    order_rule = (
        "Beat unit_id values must be in ascending source order."
        if edit_mode == "chronological"
        else "You may reorder source units when it creates a clearer montage, but preserve meaning and keep integrity_risk low."
    )
    context_rule = (
        f"Optional creator context: {content_context.strip()}"
        if content_context.strip()
        else "No extra creator context was supplied."
    )
    effects = ", ".join(sorted(ALLOWED_EFFECTS))

    return f"""
You are an expert short-form montage editor.

Create exactly {n_clips} short vertical montage plans from the supplied speech
units. Use only moments from the original source video. The hook is an on-screen
text overlay shown over the first source beat while the original audio starts
immediately; hook display time does not add to clip duration.

Infer the topic, audience, language, and editorial tone from the transcript.
Do not assume any particular subject or ideology.

Optimization goals:
- strong first seconds;
- coherent standalone idea;
- watch-through and rewatch potential;
- quotable or useful insight;
- truthful editing that does not distort the speaker's position.

Rules:
- Total spoken duration after speed adjustment must be between {min_clip_sec:.1f} and {max_clip_sec:.1f} seconds.
- Use 3 to 6 beats per clip.
- Each beat must reference an existing unit_id.
- Allowed effects: {effects}.
- speed must be between 1.0 and 1.18.
- Default to speed 1.0 unless a small acceleration improves pacing.
- overlay must be short, readable, and semantically faithful; an empty string is allowed.
- hook.text must be concise and supported by the selected material.
- hook.display_seconds should normally be 1.5 to 3.0 seconds.
- viral_score is a comparative editorial estimate from 0 to 100.
- integrity_risk is 0 to 100, where higher means the edit risks changing context or meaning.
- cta_question may be empty if a CTA would feel forced.
- Do not invent claims or quotations.
- {order_rule}

{context_rule}
""".strip()


def units_payload(units: list[Unit]) -> list[dict[str, Any]]:
    return [asdict(unit) for unit in units]
