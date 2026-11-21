from freedom_clips.prompts.opusclip import PROMPT_VERSION, build_instructions, get_clip_schema


def test_default_continuous_prompt_is_topic_and_language_neutral():
    prompt = build_instructions(
        min_clips=5,
        max_clips=10,
        min_sec=30,
        max_sec=90,
        source_language="auto",
        hook_language="same",
        editorial_style="auto",
        content_context="",
        max_overlap_sec=0.0,
    ).lower()
    banned = [
        "digital freedom",
        "censorship",
        "guru-like",
        "hindi speech",
        "hook_hi",
        "devanagari",
    ]
    for phrase in banned:
        assert phrase not in prompt


def test_continuous_prompt_requires_seconds_not_decimal_minutes():
    prompt = build_instructions(
        min_clips=5,
        max_clips=10,
        min_sec=30,
        max_sec=90,
        source_language="auto",
        hook_language="same",
        editorial_style="auto",
        content_context="",
        max_overlap_sec=0.0,
    ).lower()
    assert "absolute seconds" in prompt
    assert "00:26:54,000 becomes 1614.0" in prompt
    assert "wrong: 26.54" in prompt
    assert PROMPT_VERSION == "opusclip-generic-v3"


def test_continuous_schema_documents_seconds_only_timestamps():
    schema = get_clip_schema(5, 10)
    clip_props = schema["properties"]["clips"]["items"]["properties"]
    for field in ("start_sec", "end_sec"):
        assert clip_props[field]["minimum"] == 0
        description = clip_props[field]["description"].lower()
        assert "absolute timestamp in seconds" in description
        assert "never use mm.ss" in description
