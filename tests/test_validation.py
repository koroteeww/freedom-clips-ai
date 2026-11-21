from freedom_clips.models import Cue, Unit
from freedom_clips.validation import validate_contiguous_clips, validate_montage_plan


def test_contiguous_plan_fits_real_cue_boundaries():
    cues = [
        Cue(1, 0.0, 2.0, "one"),
        Cue(2, 2.0, 4.0, "two"),
        Cue(3, 4.0, 7.0, "three"),
        Cue(4, 7.0, 10.0, "four"),
    ]
    raw = [
        {
            "rank": 1,
            "start_sec": 0.4,
            "end_sec": 7.5,
            "viral_score": 120,
            "title": "test",
            "hook_text": "hook",
            "caption": "caption",
            "why_selected": "reason",
        }
    ]
    clips = validate_contiguous_clips(raw, cues, 10.0, 5.0, 8.0, 5, 0.0)
    assert len(clips) == 1
    assert clips[0]["start_sec"] in {cue.start for cue in cues}
    assert clips[0]["end_sec"] in {cue.end for cue in cues}
    assert clips[0]["viral_score"] == 100
    assert 5.0 <= clips[0]["duration_sec"] <= 8.0


def test_overlap_policy_is_explicit():
    cues = [Cue(i, float(i), float(i + 1), str(i)) for i in range(20)]
    raw = [
        {"rank": 1, "start_sec": 0, "end_sec": 8, "viral_score": 90, "title": "a", "hook_text": "a", "caption": "", "why_selected": ""},
        {"rank": 2, "start_sec": 5, "end_sec": 13, "viral_score": 80, "title": "b", "hook_text": "b", "caption": "", "why_selected": ""},
    ]
    clips = validate_contiguous_clips(raw, cues, 20.0, 5.0, 9.0, 5, 0.0)
    assert len(clips) == 1
    clips_with_overlap = validate_contiguous_clips(raw, cues, 20.0, 5.0, 9.0, 5, 3.0)
    assert len(clips_with_overlap) == 2


def test_decimal_minute_timestamps_are_rejected_not_snapped_to_wrong_seconds():
    cues = [Cue(i, float(i), float(i + 1), str(i)) for i in range(1800)]
    raw = [
        {
            "rank": 1,
            "start_sec": 26.54,
            "end_sec": 28.13,
            "viral_score": 90,
            "title": "bad decimal minute timestamp",
            "hook_text": "hook",
            "caption": "caption",
            "why_selected": "reason",
        }
    ]
    clips = validate_contiguous_clips(raw, cues, 1800.0, 10.0, 90.0, 5, 0.0)
    assert clips == []


def test_real_seconds_timestamps_are_still_accepted_after_one_minute():
    cues = [Cue(i, float(i), float(i + 1), str(i)) for i in range(1800)]
    raw = [
        {
            "rank": 1,
            "start_sec": 1614.0,
            "end_sec": 1693.0,
            "viral_score": 90,
            "title": "valid seconds timestamp",
            "hook_text": "hook",
            "caption": "caption",
            "why_selected": "reason",
        }
    ]
    clips = validate_contiguous_clips(raw, cues, 1800.0, 10.0, 90.0, 5, 0.0)
    assert len(clips) == 1
    assert clips[0]["start_sec"] == 1614.0
    assert clips[0]["end_sec"] == 1693.0


def test_montage_plan_duration_and_chronology():
    units = {
        0: Unit(0, 0, 5, 5, "a"),
        1: Unit(1, 5, 10, 5, "b"),
        2: Unit(2, 10, 15, 5, "c"),
    }
    raw = {
        "global_theme": "generic topic",
        "clips": [
            {
                "clip_id": 1,
                "title": "test",
                "why_selected": "reason",
                "viral_score": 88,
                "integrity_risk": 5,
                "hook": {"display_seconds": 2.0, "text": "hook"},
                "beats": [
                    {"unit_id": 0, "effect": "none", "speed": 1.0, "overlay": "", "reason": ""},
                    {"unit_id": 1, "effect": "punch_in", "speed": 1.0, "overlay": "x", "reason": ""},
                    {"unit_id": 2, "effect": "none", "speed": 1.0, "overlay": "", "reason": ""},
                ],
                "cta_question": "Thoughts?",
            }
        ],
    }
    plan = validate_montage_plan(raw, units, 15.0, 25.0, "chronological", 3)
    assert plan["clips"][0]["duration_sec"] == 15.0
