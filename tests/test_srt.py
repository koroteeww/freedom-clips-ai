from pathlib import Path

from freedom_clips.models import Cue
from freedom_clips.srt import (
    align_secondary_text,
    assess_srt_quality,
    parse_srt,
    seconds_to_ass_time,
    seconds_to_srt_time,
    write_clip_srt,
)


def test_timestamp_rounding_never_overflows_fields():
    assert seconds_to_srt_time(0) == "00:00:00,000"
    assert seconds_to_srt_time(1.9994) == "00:00:01,999"
    assert seconds_to_srt_time(1.9996) == "00:00:02,000"
    assert seconds_to_srt_time(59.9996) == "00:01:00,000"
    assert seconds_to_srt_time(3599.9996) == "01:00:00,000"
    assert seconds_to_ass_time(1.996) == "0:00:02.00"


def test_parse_numbered_and_unicode_srt(tmp_path: Path):
    path = tmp_path / "sample.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n<b>Привет мир</b>\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n你好 दुनिया\n",
        encoding="utf-8",
    )
    cues = parse_srt(path)
    assert [cue.text for cue in cues] == ["Привет мир", "你好 दुनिया"]
    assert cues[1].start == 2.0


def test_secondary_alignment_uses_time_overlap_not_index():
    primary = [
        Cue(1, 0.0, 2.0, "a"),
        Cue(2, 2.0, 4.0, "b"),
    ]
    secondary = [
        Cue(9, 1.9, 4.1, "translated b"),
        Cue(8, 0.0, 1.8, "translated a"),
    ]
    secondary = sorted(secondary, key=lambda cue: cue.start)
    aligned = align_secondary_text(primary, secondary)
    assert aligned == ["translated a", "translated b"]


def test_srt_quality_rejects_repeated_single_word_transcript():
    cues = [Cue(i, i * 5.0, i * 5.0 + 5.0, "the") for i in range(60)]

    report = assess_srt_quality(cues, min_cues=3, min_span_sec=10)

    assert not report.ok
    assert "repetitive" in report.reason
    assert report.top_word == "the"


def test_srt_quality_accepts_varied_transcript():
    cues = [
        Cue(1, 0.0, 5.0, "Here is a useful opening idea about local video editing."),
        Cue(2, 5.0, 10.0, "The workflow extracts audio, builds captions, and plans clips."),
        Cue(3, 10.0, 15.0, "Good subtitles should preserve real speech instead of hallucinated filler."),
        Cue(4, 15.0, 20.0, "Validation catches broken transcripts before the editor wastes time."),
    ]

    report = assess_srt_quality(cues, min_cues=3, min_span_sec=10)

    assert report.ok, report.reason


def test_write_clip_srt_splits_long_source_cues_into_readable_subtitles(tmp_path: Path):
    text = (
        "So, there's not one way that the United States handles the distribution of the state. "
        "So, it's, you know, 50 different states, 50 different jurisdictions. "
        "Some things are similar, but you can't know for sure how the laws are going to apply from one state to another."
    )
    out_path = tmp_path / "clip.srt"

    write_clip_srt([Cue(1, 100.0, 117.0, text)], 100.0, 117.0, out_path)

    written = out_path.read_text(encoding="utf-8")
    cues = parse_srt(out_path)
    assert len(cues) > 1
    assert cues[0].start == 0.0
    assert cues[-1].end == 17.0
    assert all(len(cue.text) <= 84 for cue in cues)
    assert "00:00:00,000 --> 00:00:17,000\n" not in written
