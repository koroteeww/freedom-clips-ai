"""Align a translated dub to the original video timeline.

Translated speech is usually longer or shorter than the source sentence.
This module decides, per subtitle line, how to absorb the difference:

1. use the pause until the next line (the dub may run into the gap);
2. if it still does not fit, speed the dub up (``atempo``) and/or slow the
   matching video slice down (``setpts``), within configurable limits;
3. trim (with a short fade) only what is left over, and report it.

Rendering is drift-free: every video slice is rendered first and its real
duration is measured with ffprobe; the audio for that slice is then cut to the
exact number of samples the measured video lasts. Errors therefore never
accumulate over a long video.

Dubs that are shorter than their slot are never slowed down; they are padded
with silence, which sounds more natural than stretched speech.
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .ffmpeg import media_duration, probe_media, run
from .srt import seconds_to_srt_time

MODES = ("both", "audio", "video", "none")
SAMPLE_RATE = 48000
EPS = 1e-6
MIN_SLOT_SEC = 0.05
FADE_SEC = 0.04


@dataclass(frozen=True)
class DubLine:
    """One dubbed line placed at ``start`` of the source timeline."""

    index: int
    start: float
    end: float
    audio: Path | None
    dub_duration: float
    text: str = ""


@dataclass
class Slot:
    """A piece of the source timeline and what happens to it."""

    src_start: float
    src_end: float
    line: DubLine | None
    video_factor: float = 1.0  # output length / source length (>= 1 slows video)
    audio_tempo: float = 1.0   # dub playback speed (>= 1 speeds dub up)
    overflow: float = 0.0      # seconds of dub (after tempo) that must be trimmed

    @property
    def src_duration(self) -> float:
        return self.src_end - self.src_start

    @property
    def out_duration(self) -> float:
        return self.src_duration * self.video_factor

    @property
    def dub_out_duration(self) -> float:
        if self.line is None:
            return 0.0
        return min(self.line.dub_duration / self.audio_tempo, self.out_duration)


@dataclass
class VideoSegment:
    """Consecutive slots rendered as one video slice with one speed factor."""

    src_start: float
    src_end: float
    video_factor: float
    slots: list[Slot] = field(default_factory=list)

    @property
    def out_duration(self) -> float:
        return (self.src_end - self.src_start) * self.video_factor


@dataclass
class AlignmentPlan:
    total_src: float
    mode: str
    max_audio_tempo: float
    max_video_slowdown: float
    segments: list[VideoSegment]
    warnings: list[str]

    @property
    def slots(self) -> list[Slot]:
        return [slot for segment in self.segments for slot in segment.slots]

    @property
    def total_out(self) -> float:
        return sum(segment.out_duration for segment in self.segments)

    @property
    def changes_video(self) -> bool:
        return any(abs(segment.video_factor - 1.0) > EPS for segment in self.segments)

    def summary(self) -> dict:
        slots = [s for s in self.slots if s.line is not None]
        return {
            "mode": self.mode,
            "lines": len(slots),
            "source_duration": round(self.total_src, 3),
            "output_duration": round(self.total_out, 3),
            "lines_sped_up": sum(1 for s in slots if s.audio_tempo > 1.0 + EPS),
            "lines_video_slowed": sum(1 for s in slots if s.video_factor > 1.0 + EPS),
            "lines_trimmed": sum(1 for s in slots if s.overflow > EPS),
            "seconds_trimmed": round(sum(s.overflow for s in slots), 3),
            "max_audio_tempo_used": round(max((s.audio_tempo for s in slots), default=1.0), 4),
            "max_video_factor_used": round(max((s.video_factor for s in slots), default=1.0), 4),
            "warnings": len(self.warnings),
        }

    def to_json(self) -> dict:
        def slot_json(slot: Slot) -> dict:
            data = {
                "src_start": round(slot.src_start, 3),
                "src_end": round(slot.src_end, 3),
                "video_factor": round(slot.video_factor, 4),
                "audio_tempo": round(slot.audio_tempo, 4),
                "overflow": round(slot.overflow, 3),
            }
            if slot.line is not None:
                data.update(
                    index=slot.line.index,
                    dub_duration=round(slot.line.dub_duration, 3),
                    audio=str(slot.line.audio) if slot.line.audio else None,
                    text=slot.line.text,
                )
            return data

        return {
            "summary": self.summary(),
            "warnings": self.warnings,
            "segments": [
                {
                    "src_start": round(seg.src_start, 3),
                    "src_end": round(seg.src_end, 3),
                    "video_factor": round(seg.video_factor, 4),
                    "slots": [slot_json(s) for s in seg.slots],
                }
                for seg in self.segments
            ],
        }


def _fit_line(available: float, dub: float, mode: str, max_tempo: float, max_slow: float) -> tuple[float, float, float]:
    """Return (video_factor, audio_tempo, overflow) for one line slot."""
    if dub <= available + EPS:
        return 1.0, 1.0, 0.0

    if mode == "none":
        return 1.0, 1.0, dub - available

    if mode == "audio":
        tempo = min(dub / available, max_tempo)
        return 1.0, tempo, max(0.0, dub / tempo - available)

    if mode == "video":
        factor = min(dub / available, max_slow)
        return factor, 1.0, max(0.0, dub - available * factor)

    # "both": share the difference between video and audio, then respect limits.
    out = (available + dub) / 2.0
    out = max(out, dub / max_tempo)      # audio may not be sped up beyond max_tempo
    out = min(out, available * max_slow)  # video may not be slowed beyond max_slow
    out = max(out, available)
    factor = out / available
    tempo = min(max(dub / out, 1.0), max_tempo)
    return factor, tempo, max(0.0, dub / tempo - out)


def plan_alignment(
    lines: Iterable[DubLine],
    total_duration: float,
    *,
    mode: str = "both",
    max_audio_tempo: float = 1.35,
    max_video_slowdown: float = 1.6,
    use_gaps: bool = True,
) -> AlignmentPlan:
    """Build an alignment plan. Pure function: no ffmpeg, no files touched."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if max_audio_tempo < 1.0 or max_video_slowdown < 1.0:
        raise ValueError("max_audio_tempo and max_video_slowdown must be >= 1.0")
    if total_duration <= 0:
        raise ValueError("total_duration must be positive")

    warnings: list[str] = []
    ordered = sorted(lines, key=lambda line: (line.start, line.index))
    kept: list[DubLine] = []
    for line in ordered:
        if line.start >= total_duration - MIN_SLOT_SEC:
            warnings.append(f"line {line.index}: starts after the end of the video, dropped")
            continue
        if kept and line.start < kept[-1].start + MIN_SLOT_SEC:
            warnings.append(
                f"line {line.index}: starts at {line.start:.3f}s, overlapping line {kept[-1].index}; dropped"
            )
            continue
        kept.append(line)

    slots: list[Slot] = []
    cursor = 0.0
    for i, line in enumerate(kept):
        start = max(0.0, line.start)
        if start > cursor + EPS:
            slots.append(Slot(cursor, start, None))
        next_start = kept[i + 1].start if i + 1 < len(kept) else total_duration
        if use_gaps:
            end = next_start
        else:
            end = min(max(line.end, start + MIN_SLOT_SEC), next_start)
        available = end - start
        factor, tempo, overflow = _fit_line(available, line.dub_duration, mode, max_audio_tempo, max_video_slowdown)
        if overflow > EPS:
            warnings.append(
                f"line {line.index}: dub {line.dub_duration:.2f}s does not fit {available:.2f}s "
                f"within limits; {overflow:.2f}s will be trimmed"
            )
        slots.append(Slot(start, end, line, factor, tempo, overflow))
        cursor = end
    if cursor < total_duration - EPS:
        slots.append(Slot(cursor, total_duration, None))

    segments: list[VideoSegment] = []
    for slot in slots:
        last = segments[-1] if segments else None
        unchanged = abs(slot.video_factor - 1.0) <= EPS
        if last is not None and unchanged and abs(last.video_factor - 1.0) <= EPS:
            last.src_end = slot.src_end
            last.slots.append(slot)
        else:
            segments.append(VideoSegment(slot.src_start, slot.src_end, slot.video_factor, [slot]))

    return AlignmentPlan(
        total_src=total_duration,
        mode=mode,
        max_audio_tempo=max_audio_tempo,
        max_video_slowdown=max_video_slowdown,
        segments=segments,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def atempo_chain(tempo: float) -> str:
    """ffmpeg ``atempo`` accepts 0.5..2.0 per instance; chain for other values."""
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    parts: list[str] = []
    remaining = tempo
    while remaining > 2.0 + EPS:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5 - EPS:
        parts.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > EPS or not parts:
        parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def _concat_line(path: Path) -> str:
    return "file '" + path.resolve().as_posix().replace("'", "'\\''") + "'"


def _video_fps(video: Path) -> float:
    for stream in probe_media(video).get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        for key in ("avg_frame_rate", "r_frame_rate"):
            num, _, den = str(stream.get(key) or "0/0").partition("/")
            try:
                fps = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                continue
            if fps > 0:
                return fps
    return 30.0


def _render_video_segment(
    video: Path, seg: VideoSegment, out: Path, fps: float, *, video_codec: str, crf: int, preset: str, verbose: bool
) -> float:
    start = round(seg.src_start * fps) / fps
    end = round(seg.src_end * fps) / fps
    duration = max(end - start, 1.0 / fps)
    vf = f"setpts={seg.video_factor:.6f}*(PTS-STARTPTS),fps={fps:.6f},format=yuv420p"
    run(
        [
            "ffmpeg", "-y", "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", str(video),
            "-an", "-sn", "-vf", vf,
            "-c:v", video_codec, "-preset", preset, "-crf", str(crf),
            str(out),
        ],
        verbose=verbose,
    )
    return media_duration(out)


def _render_slot_audio(slot: Slot, out: Path, samples: int, *, verbose: bool) -> None:
    """Write exactly ``samples`` stereo samples: the (sped-up) dub, then silence."""
    total = samples / SAMPLE_RATE
    if slot.line is None or slot.line.audio is None:
        run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
                "-af", f"atrim=end_sample={samples}", "-c:a", "pcm_s16le", str(out),
            ],
            verbose=verbose,
        )
        return
    filters = [f"aresample={SAMPLE_RATE}", "aformat=sample_fmts=s16:channel_layouts=stereo"]
    if abs(slot.audio_tempo - 1.0) > EPS:
        filters.append(atempo_chain(slot.audio_tempo))
    spoken = slot.line.dub_duration / slot.audio_tempo
    if spoken > total + EPS:
        fade_start = max(0.0, total - FADE_SEC)
        filters.append(f"afade=t=out:st={fade_start:.4f}:d={min(FADE_SEC, total):.4f}")
    filters += [f"apad=whole_len={samples}", f"atrim=end_sample={samples}", "asetpts=N/SR/TB"]
    run(
        ["ffmpeg", "-y", "-i", str(slot.line.audio), "-vn", "-af", ",".join(filters), "-c:a", "pcm_s16le", str(out)],
        verbose=verbose,
    )


def _render_background(
    source: Path, seg: VideoSegment, out: Path, samples: int, volume: float, *, verbose: bool
) -> None:
    """Background audio for one segment, time-warped exactly like the video."""
    filters = [
        f"atrim=start={seg.src_start:.6f}:end={seg.src_end:.6f}", "asetpts=PTS-STARTPTS",
        f"aresample={SAMPLE_RATE}", "aformat=sample_fmts=s16:channel_layouts=stereo",
    ]
    if abs(seg.video_factor - 1.0) > EPS:
        filters.append(atempo_chain(1.0 / seg.video_factor))
    filters += [f"volume={volume:.4f}", f"apad=whole_len={samples}", f"atrim=end_sample={samples}"]
    run(
        ["ffmpeg", "-y", "-i", str(source), "-vn", "-af", ",".join(filters), "-c:a", "pcm_s16le", str(out)],
        verbose=verbose,
    )


def _concat(files: Sequence[Path], list_path: Path, out: Path, *, verbose: bool) -> None:
    list_path.write_text("\n".join(_concat_line(p) for p in files) + "\n", encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out)], verbose=verbose)


@dataclass
class RenderResult:
    output: Path
    aligned_srt: Path
    report: Path
    output_duration: float
    timeline: list[dict]


def render_alignment(
    plan: AlignmentPlan,
    *,
    video: Path,
    out_path: Path,
    work_dir: Path,
    background: Path | None = None,
    background_volume: float = 0.2,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 20,
    preset: str = "veryfast",
    jobs: int = 2,
    keep_work: bool = False,
    verbose: bool = False,
) -> RenderResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Video: render changed slices and measure their real duration.
    if plan.changes_video:
        fps = _video_fps(video)
        seg_files = [work_dir / f"v_{i:05d}.mp4" for i in range(len(plan.segments))]
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            measured = list(
                pool.map(
                    lambda pair: _render_video_segment(
                        video, pair[0], pair[1], fps,
                        video_codec=video_codec, crf=crf, preset=preset, verbose=verbose,
                    ),
                    zip(plan.segments, seg_files),
                )
            )
        video_track = work_dir / "video.mp4"
        _concat(seg_files, work_dir / "video_list.txt", video_track, verbose=verbose)
    else:
        measured = [seg.out_duration for seg in plan.segments]
        measured[-1] += media_duration(video) - plan.total_src  # keep the original tail exactly
        video_track = video

    # 2. Audio: sample-exact per slot, boundaries derived from measured video.
    slot_files: list[Path] = []
    bg_files: list[Path] = []
    timeline: list[dict] = []
    out_cursor = 0.0
    sample_cursor = 0
    n = 0
    for seg_i, (seg, seg_measured) in enumerate(zip(plan.segments, measured)):
        nominal = seg.out_duration
        scale = seg_measured / nominal if nominal > EPS else 1.0
        seg_start_sample = sample_cursor
        for slot in seg.slots:
            slot_out = slot.out_duration * scale
            end_sample = round((out_cursor + slot_out) * SAMPLE_RATE)
            samples = max(1, end_sample - sample_cursor)
            path = work_dir / f"a_{n:05d}.wav"
            _render_slot_audio(slot, path, samples, verbose=verbose)
            slot_files.append(path)
            if slot.line is not None:
                start_t = sample_cursor / SAMPLE_RATE
                timeline.append(
                    {
                        "index": slot.line.index,
                        "start": start_t,
                        "end": start_t + min(slot.line.dub_duration / slot.audio_tempo, samples / SAMPLE_RATE),
                        "text": slot.line.text,
                    }
                )
            sample_cursor += samples
            out_cursor += slot_out
            n += 1
        if background is not None:
            bg_path = work_dir / f"b_{seg_i:05d}.wav"
            _render_background(
                background, seg, bg_path, sample_cursor - seg_start_sample, background_volume, verbose=verbose
            )
            bg_files.append(bg_path)

    dub_wav = work_dir / "dub.wav"
    _concat(slot_files, work_dir / "audio_list.txt", dub_wav, verbose=verbose)

    # 3. Mux (optionally mixing the time-warped background under the dub).
    cmd = ["ffmpeg", "-y", "-i", str(video_track), "-i", str(dub_wav)]
    if bg_files:
        bg_wav = work_dir / "background.wav"
        _concat(bg_files, work_dir / "background_list.txt", bg_wav, verbose=verbose)
        cmd += [
            "-i", str(bg_wav),
            "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[aout]",
            "-map", "0:v:0", "-map", "[aout]",
        ]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-c:v", "copy", "-c:a", audio_codec, "-b:a", "192k", "-movflags", "+faststart", str(out_path)]
    run(cmd, verbose=verbose)

    srt_path = out_path.with_suffix(".aligned.srt")
    write_timeline_srt(timeline, srt_path)
    report_path = out_path.with_suffix(".alignment.json")
    report = plan.to_json()
    report["summary"]["rendered_duration"] = round(media_duration(out_path), 3)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    return RenderResult(out_path, srt_path, report_path, media_duration(out_path), timeline)


def write_timeline_srt(timeline: Sequence[dict], out_path: Path) -> None:
    blocks = []
    for n, item in enumerate(timeline, 1):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        blocks.append(
            f"{n}\n{seconds_to_srt_time(item['start'])} --> {seconds_to_srt_time(item['end'])}\n{text}\n"
        )
    out_path.write_text("\n".join(blocks), encoding="utf-8")


def lines_from_manifest(path: Path, *, probe: bool = True) -> list[DubLine]:
    """Read a dub manifest.

    Accepts a JSON list (or ``{"items": [...]}``) of objects with ``start``,
    ``end`` and an audio path in ``audio``, ``raw_mp3`` or ``timed_mp3``
    (``raw_mp3`` is preferred: unfitted TTS gives the aligner the most room).
    ``tts_manifest.json`` from ``tretyakov_translate_tts_kdenlive.py`` works as is.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("items", []) if isinstance(data, dict) else data
    lines: list[DubLine] = []
    for n, row in enumerate(rows, 1):
        audio_value = row.get("audio") or row.get("raw_mp3") or row.get("timed_mp3")
        audio = Path(audio_value) if audio_value else None
        if audio is not None and not audio.is_absolute():
            audio = (path.parent / audio).resolve()
        if audio is not None and not audio.exists():
            raise FileNotFoundError(f"Dub audio not found for line {row.get('index', n)}: {audio}")
        duration = row.get("dub_duration") or row.get("raw_duration")
        if audio is not None and (probe or duration is None):
            duration = media_duration(audio)
        start = float(row["start"])
        lines.append(
            DubLine(
                index=int(row.get("index", n)),
                start=start,
                end=float(row.get("end", start)),
                audio=audio,
                dub_duration=float(duration or 0.0),
                text=str(row.get("text") or row.get("en") or ""),
            )
        )
    return lines


def lines_from_srt_and_dir(cues: Sequence, dub_dir: Path, texts: Sequence[str] | None = None) -> list[DubLine]:
    """Match files whose name starts with the 1-based cue number (``0001_*.mp3``)."""
    by_number: dict[int, Path] = {}
    for file in sorted(dub_dir.iterdir()):
        digits = ""
        for ch in file.name:
            if not ch.isdigit():
                break
            digits += ch
        if digits and file.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}:
            by_number.setdefault(int(digits), file)
    lines = []
    for n, cue in enumerate(cues, 1):
        audio = by_number.get(n)
        if audio is None:
            continue
        text = texts[n - 1] if texts is not None and n - 1 < len(texts) else cue.text
        lines.append(DubLine(n, cue.start, cue.end, audio, media_duration(audio), text or ""))
    return lines


__all__ = [
    "MODES",
    "AlignmentPlan",
    "DubLine",
    "RenderResult",
    "Slot",
    "VideoSegment",
    "atempo_chain",
    "lines_from_manifest",
    "lines_from_srt_and_dir",
    "plan_alignment",
    "render_alignment",
    "write_timeline_srt",
]
