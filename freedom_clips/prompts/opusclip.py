from __future__ import annotations

from typing import Any

PROMPT_VERSION = "opusclip-generic-v3"


SECONDS_TIMESTAMP_DESCRIPTION = (
    "Absolute timestamp in seconds from the beginning of the source video. "
    "Use a normal seconds value such as 93.5 for 00:01:33,500; never use MM.SS "
    "decimal-minute notation such as 1.33 or 26.54."
)


def get_clip_schema(min_clips: int, max_clips: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "analysis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "topic": {"type": "string"},
                    "source_language": {"type": "string"},
                    "audience": {"type": "string"},
                    "editorial_style": {"type": "string"},
                },
                "required": ["topic", "source_language", "audience", "editorial_style"],
            },
            "clips": {
                "type": "array",
                "minItems": min_clips,
                "maxItems": max_clips,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rank": {"type": "integer"},
                        "start_sec": {
                            "type": "number",
                            "minimum": 0,
                            "description": SECONDS_TIMESTAMP_DESCRIPTION,
                        },
                        "end_sec": {
                            "type": "number",
                            "minimum": 0,
                            "description": SECONDS_TIMESTAMP_DESCRIPTION,
                        },
                        "viral_score": {"type": "integer"},
                        "title": {"type": "string"},
                        "hook_text": {"type": "string"},
                        "caption": {"type": "string"},
                        "why_selected": {"type": "string"},
                    },
                    "required": [
                        "rank",
                        "start_sec",
                        "end_sec",
                        "viral_score",
                        "title",
                        "hook_text",
                        "caption",
                        "why_selected",
                    ],
                },
            },
        },
        "required": ["analysis", "clips"],
    }


def build_instructions(
    *,
    min_clips: int,
    max_clips: int,
    min_sec: int,
    max_sec: int,
    source_language: str,
    hook_language: str,
    editorial_style: str,
    content_context: str,
    max_overlap_sec: float,
) -> str:
    language_rule = (
        "Write hook_text in the same language as the source transcript."
        if hook_language == "same"
        else f"Write hook_text in {hook_language}."
    )
    source_rule = (
        "Infer the source language from the transcript."
        if source_language == "auto"
        else f"Treat the source language as {source_language}."
    )
    style_rule = (
        "Infer the most appropriate editorial tone from the content itself."
        if editorial_style == "auto"
        else f"Use this editorial style when it fits the source meaning: {editorial_style}."
    )
    context_rule = (
        f"Optional creator context: {content_context.strip()}"
        if content_context.strip()
        else "No extra creator context was supplied."
    )

    return f"""
You are an expert short-form video editor and transcript analyst.

Analyze the supplied timecoded transcript and select the strongest standalone
continuous clips for TikTok, Instagram Reels, YouTube Shorts, LinkedIn video,
and other vertical short-form platforms.

Do not assume any particular subject, ideology, country, industry, profession,
audience, or content style. Infer topic, audience, source language, and suitable
editorial tone from the transcript itself.

Selection priorities:
- a strong opening within the selected source range;
- standalone clarity without missing context;
- emotional or intellectual tension when it genuinely exists;
- novelty, quotability, useful insight, surprise, story, or clear payoff;
- faithful representation of the speaker's meaning;
- natural sentence and thought boundaries.

Rules:
- Return between {min_clips} and {max_clips} candidates.
- Each candidate must be between {min_sec} and {max_sec} seconds.
- Every candidate is one contiguous range from the original video.
- Use timestamps from the supplied transcript, converted to absolute seconds from video start.
- start_sec and end_sec MUST be real seconds, not minutes and not SRT strings.
- Convert HH:MM:SS,mmm exactly: 00:01:33,500 becomes 93.5, 00:26:54,000 becomes 1614.0.
- Never encode minute timestamps as decimal numbers. Wrong: 1.33 for 00:01:33, wrong: 26.54 for 00:26:54.
- If the chosen transcript line starts after the first minute, start_sec must be greater than 60.
- Avoid greetings, housekeeping, repeated material, and weak setup when stronger material exists.
- Do not invent facts, claims, quotations, or context that is not supported by the transcript.
- Score viral_score from 0 to 100 as a comparative editorial estimate, not a promise of performance.
- Prefer no overlap. Maximum tolerated overlap between final selections is {max_overlap_sec:.2f} seconds.
- Keep hook_text concise and truthful.
- caption should summarize the clip without adding unsupported claims.
- why_selected should explain the editorial reason briefly.

{source_rule}
{language_rule}
{style_rule}
{context_rule}
""".strip()
