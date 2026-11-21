from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from freedom_clips.dub_align import (
    DubLine,
    atempo_chain,
    lines_from_manifest,
    plan_alignment,
    render_alignment,
)
from freedom_clips.ffmpeg import probe_media
from freedom_clips.srt import parse_srt


def line(index: int, start: float, end: float, dub: float, text: str = "") -> DubLine:
    return DubLine(index, start, end, None, dub, text)


def line_slots(plan):
    return [s for s in plan.slots if s.line is not None]


def test_short_dub_is_never_stretched_and_timeline_is_unchanged():
    plan = plan_alignment([line(1, 1.0, 3.0, 1.2), line(2, 4.0, 6.0, 0.8)], 10.0)
    assert plan.total_out == pytest.approx(10.0)
    assert not plan.changes_video
    assert all(s.audio_tempo == 1.0 and s.overflow == 0 for s in line_slots(plan))
    assert len(plan.segments) == 1  # unchanged slots merge into one slice


def test_gap_to_next_line_absorbs_overrun():
    # Cue is 2s, but the next line starts at 5s, so a 2.8s dub fits without changes.
    plan = plan_alignment([line(1, 1.0, 3.0, 2.8), line(2, 5.0, 6.0, 0.5)], 8.0)
    first = line_slots(plan)[0]
    assert first.video_factor == 1.0 and first.audio_tempo == 1.0
    assert plan.total_out == pytest.approx(8.0)


def test_no_use_gaps_forces_adjustment_and_keeps_pause_slot():
    plan = plan_alignment([line(1, 1.0, 3.0, 2.8), line(2, 5.0, 6.0, 0.5)], 8.0, use_gaps=False, mode="audio")
    first = line_slots(plan)[0]
    assert first.audio_tempo == pytest.approx(1.35)  # capped by the default limit
    starts = [round(s.src_start, 3) for s in plan.slots]
    assert 3.0 in starts  # the pause stays its own slot


def test_mode_both_splits_difference_between_audio_and_video():
    plan = plan_alignment([line(1, 0.0, 2.0, 3.0)], 2.0, mode="both", max_audio_tempo=2, max_video_slowdown=2)
    slot = line_slots(plan)[0]
    assert slot.out_duration == pytest.approx(2.5)
    assert slot.video_factor == pytest.approx(1.25)
    assert slot.audio_tempo == pytest.approx(1.2)
    assert slot.dub_out_duration == pytest.approx(slot.out_duration)
    assert slot.overflow == 0


def test_mode_both_respects_limits_and_reports_overflow():
    plan = plan_alignment([line(1, 0.0, 2.0, 6.0)], 2.0, mode="both", max_audio_tempo=1.25, max_video_slowdown=1.5)
    slot = line_slots(plan)[0]
    assert slot.video_factor == pytest.approx(1.5)
    assert slot.audio_tempo == pytest.approx(1.25)
    assert slot.overflow == pytest.approx(6.0 / 1.25 - 3.0)
    assert plan.warnings and "trimmed" in plan.warnings[0]


def test_mode_both_prefers_less_video_change_when_audio_limit_allows():
    # Small overrun: the midpoint already satisfies both limits.
    plan = plan_alignment([line(1, 0.0, 4.0, 4.4)], 4.0)
    slot = line_slots(plan)[0]
    assert 1.0 < slot.video_factor < 1.1
    assert 1.0 < slot.audio_tempo < 1.1


@pytest.mark.parametrize("mode", ["audio", "video", "none"])
def test_single_strategy_modes(mode):
    plan = plan_alignment([line(1, 0.0, 2.0, 2.4)], 2.0, mode=mode)
    slot = line_slots(plan)[0]
    if mode == "audio":
        assert slot.video_factor == 1.0 and slot.audio_tempo == pytest.approx(1.2)
    elif mode == "video":
        assert slot.audio_tempo == 1.0 and slot.video_factor == pytest.approx(1.2)
    else:
        assert slot.video_factor == 1.0 and slot.audio_tempo == 1.0
        assert slot.overflow == pytest.approx(0.4)


def test_lead_in_and_tail_are_kept_and_overlaps_dropped():
    plan = plan_alignment([line(1, 2.0, 3.0, 0.5), line(2, 2.01, 3.0, 0.5), line(3, 6.0, 7.0, 0.5)], 9.0, use_gaps=False)
    assert plan.slots[0].line is None and plan.slots[0].src_start == 0.0
    assert plan.slots[-1].line is None and plan.slots[-1].src_end == 9.0
    assert [s.line.index for s in line_slots(plan)] == [1, 3]
    assert any("line 2" in w for w in plan.warnings)
    covered = sum(s.src_duration for s in plan.slots)
    assert covered == pytest.approx(9.0)


def test_slowed_slots_get_their_own_video_segments():
    lines = [line(1, 0.0, 2.0, 1.0), line(2, 2.0, 4.0, 3.0), line(3, 4.0, 6.0, 1.0), line(4, 6.0, 8.0, 1.0)]
    plan = plan_alignment(lines, 8.0)
    factors = [round(seg.video_factor, 3) for seg in plan.segments]
    assert len(plan.segments) == 3
    assert factors[0] == 1.0 and factors[1] > 1.0 and factors[2] == 1.0
    assert len(plan.segments[2].slots) == 2


def test_invalid_arguments():
    with pytest.raises(ValueError):
        plan_alignment([], 5.0, mode="fast")
    with pytest.raises(ValueError):
        plan_alignment([], 5.0, max_audio_tempo=0.9)


def test_atempo_chain_stays_in_ffmpeg_range():
    assert atempo_chain(1.0) == "atempo=1.000000"
    assert atempo_chain(3.0) == "atempo=2.0,atempo=1.500000"
    assert atempo_chain(0.3) == "atempo=0.5,atempo=0.600000"


def test_manifest_reader_accepts_tretyakov_format(tmp_path: Path):
    audio = tmp_path / "raw" / "0001_A.mp3"
    audio.parent.mkdir()
    audio.write_bytes(b"")
    manifest = tmp_path / "tts_manifest.json"
    manifest.write_text(
        json.dumps([{"index": 7, "start": 1.5, "end": 3.0, "raw_mp3": "raw/0001_A.mp3", "raw_duration": 1.9, "en": "Hi"}]),
        encoding="utf-8",
    )
    [item] = lines_from_manifest(manifest, probe=False)
    assert item.index == 7 and item.start == 1.5 and item.dub_duration == 1.9
    assert item.audio == audio.resolve() and item.text == "Hi"


# --------------------------------------------------------------------------
# ffmpeg integration: A/V must stay in sync after slicing and speed changes.
# --------------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed"
)


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stream_durations(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for stream in probe_media(path)["streams"]:
        if stream.get("duration"):
            out[stream["codec_type"]] = float(stream["duration"])
    return out


def _make_inputs(tmp_path: Path) -> tuple[Path, list[DubLine]]:
    video = tmp_path / "source.mp4"
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000",
        "-t", "8", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(video),
    )
    specs = [(1, 0.5, 2.0, 1.0), (2, 2.0, 4.0, 3.2), (3, 4.0, 6.0, 1.5), (4, 6.5, 8.0, 2.5)]
    lines = []
    for index, start, end, dur in specs:
        wav = tmp_path / f"{index:04d}.wav"
        _ff("-f", "lavfi", "-i", f"sine=frequency={300 + 100 * index}:sample_rate=24000", "-t", str(dur), str(wav))
        lines.append(DubLine(index, start, end, wav, dur, f"line {index}"))
    return video, lines


@needs_ffmpeg
@pytest.mark.parametrize("mode", ["both", "video"])
def test_render_keeps_audio_and_video_in_sync(tmp_path: Path, mode: str):
    video, lines = _make_inputs(tmp_path)
    plan = plan_alignment(lines, 8.0, mode=mode, max_audio_tempo=1.3, max_video_slowdown=1.5)
    assert plan.changes_video
    result = render_alignment(
        plan, video=video, out_path=tmp_path / "out.mp4", work_dir=tmp_path / "work",
        background=video, background_volume=0.1, crf=30, preset="ultrafast",
    )
    durations = _stream_durations(result.output)
    assert abs(durations["video"] - plan.total_out) < 0.15
    assert abs(durations["audio"] - durations["video"]) < 0.06  # no drift between tracks

    cues = parse_srt(result.aligned_srt)
    assert [c.text for c in cues] == ["line 1", "line 2", "line 3", "line 4"]
    assert cues[0].start == pytest.approx(0.5, abs=0.01)
    assert all(a.end <= b.start + 1e-3 for a, b in zip(cues, cues[1:]))
    report = json.loads(result.report.read_text(encoding="utf-8"))
    assert report["summary"]["lines"] == 4
    assert not (tmp_path / "work").exists()


@needs_ffmpeg
def test_audio_only_mode_copies_video_and_keeps_length(tmp_path: Path):
    video, lines = _make_inputs(tmp_path)
    plan = plan_alignment(lines, 8.0, mode="audio")
    assert not plan.changes_video
    result = render_alignment(plan, video=video, out_path=tmp_path / "out.mp4", work_dir=tmp_path / "work")
    durations = _stream_durations(result.output)
    source = _stream_durations(video)
    assert abs(durations["video"] - source["video"]) < 0.01
    assert abs(durations["audio"] - durations["video"]) < 0.06
