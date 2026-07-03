#!/usr/bin/env python3
"""
build_exact_dub_mp3_from_manifest.py

Purpose:
  Build ONE full English dub MP3 from:
    - translated_en.json   (timeline source: index/start/end/speaker/en)
    - original SRT         (fallback timeline source)
    - tts_manifest.json    (audio source: timed_mp3 per index)

Why this exists:
  Simple concat of audio_timed/*.mp3 is wrong for a full film:
    - every MP3 has encoder padding / duration drift
    - small overshoots accumulate into minutes
    - SRT fragments can overlap
    - there are gaps/music/no-speech parts that must remain silence

Correct approach here:
  1) Read timeline from translated_en.json / SRT.
  2) Read timed_mp3 paths from tts_manifest.json.
  3) Normalize every spoken MP3 to exact target duration as WAV.
  4) Render fixed-length timeline chunks by placing each fragment at its SRT start time.
  5) Mix overlaps if they exist.
  6) Concatenate chunk WAVs.
  7) Encode final MP3.

Install:
  ffmpeg and ffprobe must be in PATH.
  No OpenAI API is needed.

Example:
  python build_exact_dub_mp3_from_manifest.py ^
    --translated_json "D:\GLOBAL_Rayban_Meta\_tretyakov\eng2\translated_en.json" ^
    --srt "D:\GLOBAL_Rayban_Meta\full--transcript.srt" ^
    --tts_manifest "D:\GLOBAL_Rayban_Meta\_tretyakov\eng2\tts_manifest.json" ^
    --outdir "D:\GLOBAL_Rayban_Meta\_tretyakov\eng2\exact_join" ^
    --video "D:\GLOBAL_Rayban_Meta\tretyakov.mp4" ^
    --pad_to_video_length

If you only want to end at last spoken subtitle:
  omit --pad_to_video_length

Outputs:
  tretyakov_english_dub_full_exact.mp3
  tretyakov_english_dub_full_exact.wav   unless --delete_full_wav
  join_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TimelineItem:
    index: int
    start: float
    end: float
    speaker: str
    en: str = ""
    ru: str = ""


@dataclass
class ManifestItem:
    index: int
    speaker: str
    start: float
    end: float
    target_duration: float
    timed_mp3: str
    final_duration: Optional[float] = None
    raw_duration: Optional[float] = None
    en: str = ""


@dataclass
class JoinItem:
    index: int
    speaker: str
    start: float
    end: float
    target_duration: float
    source_mp3: str
    exact_wav: str
    exists: bool
    note: str = ""


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_tool(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} is not in PATH. Install it or add it to PATH.")


def run(cmd: List[str]) -> None:
    pretty = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    log(f"[CMD] {pretty}")
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def clean_text(text: str) -> str:
    text = str(text or "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def safe_file_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s or ""))
    return s.strip("_") or "speaker"


def srt_time_to_sec(s: str) -> float:
    h, m, rest = s.strip().split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def sec_to_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    if ms >= 1000:
        total += 1
        ms -= 1000
    s = total % 60
    total //= 60
    m = total % 60
    h = total // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ffmpeg_concat_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    s = s.replace("'", "'\\''")
    return s


def parse_srt(path: Path) -> Dict[int, TimelineItem]:
    if not path or not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")
    result: Dict[int, TimelineItem] = {}

    for fallback_i, block in enumerate(blocks, 1):
        lines = [x.strip() for x in block.split("\n") if x.strip()]
        if len(lines) < 2:
            continue

        try:
            idx = int(lines[0])
            time_line = lines[1]
            text_lines = lines[2:]
        except ValueError:
            idx = fallback_i
            time_line = lines[0]
            text_lines = lines[1:]

        m = time_re.search(time_line)
        if not m:
            continue

        text = clean_text(" ".join(text_lines))
        speaker = "SPEAKER_UNKNOWN"
        sm = re.match(r"^\[([^\]]+)\]\s*(.*)$", text)
        if sm:
            speaker = sm.group(1).strip()
            text = clean_text(sm.group(2))

        result[idx] = TimelineItem(
            index=idx,
            start=srt_time_to_sec(m.group(1)),
            end=srt_time_to_sec(m.group(2)),
            speaker=speaker,
            ru=text,
        )

    return result


def load_translated_json(path: Path) -> Dict[int, TimelineItem]:
    if not path or not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("items", data if isinstance(data, list) else [])
    result: Dict[int, TimelineItem] = {}

    for row in rows:
        try:
            idx = int(row["index"])
            start = float(row["start"])
            end = float(row["end"])
        except Exception:
            continue

        result[idx] = TimelineItem(
            index=idx,
            start=start,
            end=end,
            speaker=str(row.get("speaker", "SPEAKER_UNKNOWN")),
            en=str(row.get("en", "")),
            ru=str(row.get("ru", "")),
        )

    return result


def load_manifest(path: Path) -> Dict[int, ManifestItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[int, ManifestItem] = {}

    for row in data:
        try:
            idx = int(row["index"])
            start = float(row.get("start", 0.0))
            end = float(row.get("end", start + float(row.get("target_duration", 0.0))))
            target = float(row.get("target_duration", max(0.001, end - start)))
            timed = str(row["timed_mp3"])
        except Exception:
            continue

        result[idx] = ManifestItem(
            index=idx,
            speaker=str(row.get("speaker", "SPEAKER_UNKNOWN")),
            start=start,
            end=end,
            target_duration=target,
            timed_mp3=timed,
            final_duration=float(row["final_duration"]) if row.get("final_duration") is not None else None,
            raw_duration=float(row["raw_duration"]) if row.get("raw_duration") is not None else None,
            en=str(row.get("en", "")),
        )

    return result


def build_join_items(
    translated: Dict[int, TimelineItem],
    srt_items: Dict[int, TimelineItem],
    manifest: Dict[int, ManifestItem],
    exact_dir: Path,
    prefer_json_timeline: bool = True,
) -> List[JoinItem]:
    ensure_dir(exact_dir)

    join_items: List[JoinItem] = []

    for idx, mi in sorted(manifest.items()):
        ti = None
        if prefer_json_timeline:
            ti = translated.get(idx) or srt_items.get(idx)
        else:
            ti = srt_items.get(idx) or translated.get(idx)

        if ti:
            start = float(ti.start)
            end = float(ti.end)
            speaker = ti.speaker or mi.speaker
        else:
            start = float(mi.start)
            end = float(mi.end)
            speaker = mi.speaker

        target = max(0.050, end - start)
        source = Path(mi.timed_mp3)
        exact_wav = exact_dir / f"{idx:04d}_{safe_file_part(speaker)}.wav"

        exists = source.exists()
        note = ""
        if not exists:
            note = f"missing source mp3: {source}"

        join_items.append(JoinItem(
            index=idx,
            speaker=speaker,
            start=start,
            end=end,
            target_duration=target,
            source_mp3=str(source),
            exact_wav=str(exact_wav),
            exists=exists,
            note=note,
        ))

    return join_items


def normalize_to_exact_wav(
    item: JoinItem,
    sample_rate: int,
    force: bool = False,
) -> None:
    source = Path(item.source_mp3)
    out = Path(item.exact_wav)

    if not item.exists:
        return

    if out.exists() and not force:
        return

    ensure_dir(out.parent)
    dur = max(0.050, float(item.target_duration))

    # Important:
    # - apad pads if TTS is shorter
    # - atrim cuts if TTS is longer
    # - output is PCM WAV to eliminate MP3 encoder padding before timeline mix
    run([
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",
        "-af", f"aresample={sample_rate},aformat=sample_fmts=s16:channel_layouts=stereo,apad,atrim=0:{dur:.3f},asetpts=PTS-STARTPTS",
        "-ac", "2",
        "-ar", str(sample_rate),
        "-codec:a", "pcm_s16le",
        str(out),
    ])


def create_silence_wav(out_path: Path, duration: float, sample_rate: int) -> None:
    ensure_dir(out_path.parent)
    duration = max(0.050, float(duration))

    if out_path.exists():
        return

    run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-t", f"{duration:.3f}",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
        "-codec:a", "pcm_s16le",
        "-ac", "2",
        "-ar", str(sample_rate),
        str(out_path),
    ])


def render_chunk_wav(
    chunk_index: int,
    chunk_start: float,
    chunk_end: float,
    items: List[JoinItem],
    out_wav: Path,
    sample_rate: int,
    force: bool = False,
) -> None:
    ensure_dir(out_wav.parent)
    chunk_duration = max(0.050, chunk_end - chunk_start)

    if out_wav.exists() and not force:
        return

    active: List[JoinItem] = []
    for item in items:
        if not item.exists:
            continue

        item_start = float(item.start)
        item_end = float(item.start + item.target_duration)

        # Include if it intersects this chunk window.
        if item_start < chunk_end and item_end > chunk_start:
            active.append(item)

    if not active:
        log(f"[CHUNK {chunk_index:04d}] silence only {sec_to_time(chunk_start)} - {sec_to_time(chunk_end)}")
        create_silence_wav(out_wav, chunk_duration, sample_rate)
        return

    log(f"[CHUNK {chunk_index:04d}] active={len(active)} {sec_to_time(chunk_start)} - {sec_to_time(chunk_end)}")

    cmd: List[str] = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-t", f"{chunk_duration:.3f}",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
    ]

    # Inputs start at index 1 because 0 is silence base.
    for item in active:
        cmd += ["-i", str(Path(item.exact_wav))]

    filters: List[str] = [
        f"[0:a]atrim=0:{chunk_duration:.3f},asetpts=PTS-STARTPTS[a0]"
    ]

    input_labels = ["[a0]"]

    for local_i, item in enumerate(active, 1):
        item_start = float(item.start)
        item_end = float(item.start + item.target_duration)

        # If item began before this chunk, trim its beginning.
        trim_start = max(0.0, chunk_start - item_start)

        # Use only the part that fits in this chunk.
        visible_start = max(item_start, chunk_start)
        visible_end = min(item_end, chunk_end)
        visible_duration = max(0.050, visible_end - visible_start)

        # Delay relative to chunk start.
        delay_sec = max(0.0, item_start - chunk_start)
        delay_ms = int(round(delay_sec * 1000))

        label = f"a{local_i}"
        filters.append(
            f"[{local_i}:a]"
            f"atrim=start={trim_start:.3f}:duration={visible_duration:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"adelay={delay_ms}|{delay_ms}"
            f"[{label}]"
        )
        input_labels.append(f"[{label}]")

    filters.append(
        "".join(input_labels)
        + f"amix=inputs={len(input_labels)}:duration=first:dropout_transition=0:normalize=0,"
        + f"atrim=0:{chunk_duration:.3f},asetpts=PTS-STARTPTS[out]"
    )

    filter_complex = ";".join(filters)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-codec:a", "pcm_s16le",
        "-ac", "2",
        "-ar", str(sample_rate),
        str(out_wav),
    ]

    run(cmd)


def concat_wavs(chunk_wavs: List[Path], out_wav: Path) -> None:
    ensure_dir(out_wav.parent)

    concat_list = out_wav.with_suffix(".concat.txt")
    concat_list.write_text(
        "\n".join([f"file '{ffmpeg_concat_path(p)}'" for p in chunk_wavs]) + "\n",
        encoding="utf-8",
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-codec:a", "pcm_s16le",
        str(out_wav),
    ])


def encode_mp3(input_wav: Path, out_mp3: Path, bitrate: str = "192k") -> None:
    ensure_dir(out_mp3.parent)

    run([
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-vn",
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_mp3),
    ])


def build_exact_full_dub(
    translated_json: Path,
    srt_path: Optional[Path],
    tts_manifest: Path,
    outdir: Path,
    output_mp3: Path,
    video: Optional[Path] = None,
    pad_to_video_length: bool = False,
    total_duration: Optional[float] = None,
    chunk_seconds: float = 300.0,
    sample_rate: int = 48000,
    bitrate: str = "192k",
    force: bool = False,
    keep_work: bool = True,
) -> Dict[str, Any]:
    require_tool("ffmpeg")
    require_tool("ffprobe")

    ensure_dir(outdir)

    translated = load_translated_json(translated_json)
    srt_items = parse_srt(srt_path) if srt_path else {}
    manifest = load_manifest(tts_manifest)

    exact_dir = outdir / "exact_wav_fragments"
    chunks_dir = outdir / "timeline_chunks_wav"
    ensure_dir(exact_dir)
    ensure_dir(chunks_dir)

    join_items = build_join_items(translated, srt_items, manifest, exact_dir)

    if not join_items:
        raise RuntimeError("No join items loaded from manifest")

    missing = [x for x in join_items if not x.exists]
    if missing:
        log(f"[WARN] Missing source MP3 files: {len(missing)}. They will be silent gaps.")

    # Determine timeline duration.
    last_speech_end = max(x.start + x.target_duration for x in join_items)

    if total_duration is None and pad_to_video_length and video and video.exists():
        total_duration = ffprobe_duration(video)

    if total_duration is None:
        total_duration = last_speech_end

    total_duration = max(total_duration, last_speech_end)
    log(f"[TIMELINE] last_speech_end={last_speech_end:.3f}s ({sec_to_time(last_speech_end)})")
    log(f"[TIMELINE] total_duration={total_duration:.3f}s ({sec_to_time(total_duration)})")

    # Step 1: exact WAV fragments.
    for n, item in enumerate(join_items, 1):
        if not item.exists:
            continue
        log(f"[EXACT {n}/{len(join_items)}] {item.index:04d} target={item.target_duration:.3f}s")
        normalize_to_exact_wav(item, sample_rate=sample_rate, force=force)

    # Step 2: render timeline chunks.
    chunk_wavs: List[Path] = []
    num_chunks = int(math.ceil(total_duration / chunk_seconds))

    for ci in range(num_chunks):
        start = ci * chunk_seconds
        end = min(total_duration, (ci + 1) * chunk_seconds)
        out_wav = chunks_dir / f"chunk_{ci:04d}_{start:.3f}_{end:.3f}.wav"
        render_chunk_wav(
            chunk_index=ci,
            chunk_start=start,
            chunk_end=end,
            items=join_items,
            out_wav=out_wav,
            sample_rate=sample_rate,
            force=force,
        )
        chunk_wavs.append(out_wav)

    # Step 3: concat chunks to full WAV.
    full_wav = outdir / "tretyakov_english_dub_full_exact.wav"
    concat_wavs(chunk_wavs, full_wav)

    # Step 4: encode MP3.
    encode_mp3(full_wav, output_mp3, bitrate=bitrate)

    final_wav_duration = ffprobe_duration(full_wav)
    final_mp3_duration = ffprobe_duration(output_mp3)

    report = {
        "translated_json": str(translated_json),
        "srt": str(srt_path) if srt_path else "",
        "tts_manifest": str(tts_manifest),
        "video": str(video) if video else "",
        "output_mp3": str(output_mp3),
        "full_wav": str(full_wav),
        "count_manifest": len(manifest),
        "count_join_items": len(join_items),
        "count_missing_mp3": len(missing),
        "last_speech_end_sec": last_speech_end,
        "total_duration_sec": total_duration,
        "final_wav_duration_sec": final_wav_duration,
        "final_mp3_duration_sec": final_mp3_duration,
        "diff_wav_minus_expected_sec": final_wav_duration - total_duration,
        "diff_mp3_minus_expected_sec": final_mp3_duration - total_duration,
        "chunk_seconds": chunk_seconds,
        "sample_rate": sample_rate,
        "pad_to_video_length": pad_to_video_length,
        "missing": [asdict(x) for x in missing[:50]],
    }

    report_path = outdir / "join_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log("[DONE]")
    log(f"MP3: {output_mp3}")
    log(f"WAV duration: {final_wav_duration:.3f}s, diff={final_wav_duration - total_duration:.3f}s")
    log(f"MP3 duration: {final_mp3_duration:.3f}s, diff={final_mp3_duration - total_duration:.3f}s")
    log(f"Report: {report_path}")

    if not keep_work:
        # Keep final MP3 and report; delete heavy intermediate WAVs.
        if full_wav.exists():
            full_wav.unlink()
        shutil.rmtree(exact_dir, ignore_errors=True)
        shutil.rmtree(chunks_dir, ignore_errors=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translated_json", required=True, type=Path)
    parser.add_argument("--srt", type=Path, default=None)
    parser.add_argument("--tts_manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--output_mp3", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--pad_to_video_length", action="store_true")
    parser.add_argument("--total_duration", type=float, default=None, help="Optional exact timeline duration in seconds")
    parser.add_argument("--chunk_seconds", type=float, default=300.0, help="Timeline render chunk size. 300 sec is safe on Windows.")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--bitrate", default="192k")
    parser.add_argument("--force", action="store_true", help="Rebuild exact WAV fragments/chunks even if already exist")
    parser.add_argument("--delete_work", action="store_true", help="Delete heavy intermediate WAV folders after MP3 is created")
    args = parser.parse_args()

    output_mp3 = args.output_mp3 or (args.outdir / "tretyakov_english_dub_full_exact.mp3")

    build_exact_full_dub(
        translated_json=args.translated_json,
        srt_path=args.srt,
        tts_manifest=args.tts_manifest,
        outdir=args.outdir,
        output_mp3=output_mp3,
        video=args.video,
        pad_to_video_length=args.pad_to_video_length,
        total_duration=args.total_duration,
        chunk_seconds=args.chunk_seconds,
        sample_rate=args.sample_rate,
        bitrate=args.bitrate,
        force=args.force,
        keep_work=not args.delete_work,
    )


if __name__ == "__main__":
    main()
