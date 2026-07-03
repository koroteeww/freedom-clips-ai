#!/usr/bin/env python3
"""
transcribe_ru_diarized_srt_openai.py

MP4/MOV/MKV/audio -> Russian diarized SRT using OpenAI gpt-4o-transcribe-diarize.

Install:
  pip install --upgrade openai
  ffmpeg + ffprobe must be in PATH

Environment:
  set OPENAI_API_KEY=sk-...
  PowerShell:
  $env:OPENAI_API_KEY="sk-..."

Example:
  python transcribe_ru_diarized_srt_openai.py ^
    --input "D:\GLOBAL_Rayban_Meta\tretyakov.mp4" ^
    --outdir "D:\GLOBAL_Rayban_Meta\_tretyakov_asr" ^
    --language ru

Outputs:
  tretyakov.ru.diarized.srt
  tretyakov.ru.diarized.json
  tretyakov.ru.diarized.txt
  tretyakov.openai_raw.json
  work/audio_16k_32k.mp3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


MAX_UPLOAD_MB_DEFAULT = 24.0
DEFAULT_CHUNK_SECONDS = 20 * 60


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str
    chunk_index: int = 0


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: List[str]) -> None:
    pretty = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    log(f"[CMD] {pretty}")
    subprocess.run(cmd, check=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_tool(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} is not in PATH. Install it or add it to PATH.")


def file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def sec_to_srt_time(sec: float) -> str:
    if sec < 0:
        sec = 0
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    if ms >= 1000:
        total += 1
        ms -= 1000
    s = total % 60
    total //= 60
    m = total % 60
    h = total // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


HALLUCINATION_PATTERNS = [
    r"редактор\s+субтитров",
    r"субтитр[ыо]?\s+(сделал|сделала|создал|создала)",
    r"корректор\s+субтитров",
    r"перевод\s+субтитров",
    r"расшифровка\s+.+",
    r"продолжение\s+следует",
    r"спасибо\s+за\s+просмотр",
    r"подписывайтесь\s+на\s+канал",
]


def looks_like_hallucination(text: str) -> bool:
    t = clean_text(text).lower()
    if not t:
        return True

    for pat in HALLUCINATION_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return True

    if len(t) < 45 and any(x in t for x in ["субтитр", "редактор", "корректор", "тайминг"]):
        return True

    return False


def extract_audio_mp3(input_video: Path, out_mp3: Path, bitrate: str = "32k") -> None:
    ensure_dir(out_mp3.parent)
    run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", bitrate,
        str(out_mp3),
    ])


def split_audio(input_audio: Path, chunks_dir: Path, chunk_seconds: int) -> List[Path]:
    ensure_dir(chunks_dir)
    pattern = chunks_dir / "chunk_%04d.mp3"

    run([
        "ffmpeg", "-y",
        "-i", str(input_audio),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        "-c", "copy",
        str(pattern),
    ])

    chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("ffmpeg produced no chunks")
    return chunks


def response_to_dict(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))


def data_url_for_audio(path: Path) -> str:
    import base64
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lower().lstrip(".") or "wav"
    mime = "audio/wav" if ext == "wav" else "audio/mpeg"
    return f"data:{mime};base64,{data}"


def call_openai_diarize(
    client: OpenAI,
    audio_path: Path,
    language: str = "ru",
    known_speaker_names: Optional[List[str]] = None,
    known_speaker_reference_paths: Optional[List[Path]] = None,
    retries: int = 3,
) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}

    if known_speaker_names and known_speaker_reference_paths:
        extra_body["known_speaker_names"] = known_speaker_names
        extra_body["known_speaker_references"] = [
            data_url_for_audio(Path(p)) for p in known_speaker_reference_paths
        ]

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            log(f"[OPENAI] Transcribing {audio_path.name} ({file_mb(audio_path):.2f} MB), attempt {attempt}/{retries}")
            with audio_path.open("rb") as f:
                kwargs: Dict[str, Any] = {
                    "model": "gpt-4o-transcribe-diarize",
                    "file": f,
                    "response_format": "diarized_json",
                    "chunking_strategy": "auto",
                }

                if language:
                    kwargs["language"] = language

                if extra_body:
                    kwargs["extra_body"] = extra_body

                resp = client.audio.transcriptions.create(**kwargs)

            d = response_to_dict(resp)
            if "segments" not in d:
                log("[WARN] Response has no 'segments'. Raw keys: " + ", ".join(d.keys()))
            return d

        except TypeError as e:
            log(f"[ERROR] SDK TypeError: {e}")
            log("Try: pip install --upgrade openai")
            raise
        except Exception as e:
            last_error = e
            log(f"[WARN] OpenAI failed attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)

    raise RuntimeError(f"OpenAI transcription failed after {retries} retries: {last_error}")


def parse_segments(raw: Dict[str, Any], chunk_index: int, offset: float, remove_hallucinations: bool) -> List[Segment]:
    result: List[Segment] = []
    raw_segments = raw.get("segments") or []

    if not raw_segments and raw.get("text"):
        text = clean_text(str(raw["text"]))
        if not (remove_hallucinations and looks_like_hallucination(text)):
            result.append(Segment(start=offset, end=offset + 1.0, speaker="SPEAKER_00", text=text, chunk_index=chunk_index))
        return result

    for s in raw_segments:
        if not isinstance(s, dict):
            s = response_to_dict(s)

        text = clean_text(str(s.get("text", "")))
        if remove_hallucinations and looks_like_hallucination(text):
            log(f"[FILTER] Removed hallucinated line: {text}")
            continue

        try:
            start = float(s.get("start", 0.0)) + offset
            end = float(s.get("end", start + 1.0)) + offset
        except Exception:
            start = offset
            end = offset + 1.0

        speaker = str(s.get("speaker", "") or s.get("speaker_id", "") or "SPEAKER_00").strip()
        if not speaker:
            speaker = "SPEAKER_00"

        speaker = speaker.replace(" ", "_")
        if re.fullmatch(r"[A-Z]", speaker):
            speaker = f"SPEAKER_{speaker}"
        elif re.fullmatch(r"\d+", speaker):
            speaker = f"SPEAKER_{int(speaker):02d}"

        if text:
            result.append(Segment(start=start, end=end, speaker=speaker, text=text, chunk_index=chunk_index))

    return result


def split_long_segment_text(text: str, max_chars: int = 130) -> List[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]

    pieces = re.split(r"(?<=[.!?…])\s+", text)
    lines: List[str] = []
    cur = ""

    for p in pieces:
        if not p:
            continue
        if len(cur) + 1 + len(p) <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                lines.append(cur)
            cur = p

    if cur:
        lines.append(cur)

    final: List[str] = []
    for line in lines or [text]:
        if len(line) <= max_chars:
            final.append(line)
            continue
        words = line.split()
        cur = ""
        for w in words:
            if len(cur) + 1 + len(w) <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    final.append(cur)
                cur = w
        if cur:
            final.append(cur)

    return final


def write_srt(segments: List[Segment], out_path: Path, split_long: bool = True, include_speaker: bool = True) -> None:
    lines: List[str] = []
    idx = 1

    for seg in segments:
        if seg.end <= seg.start:
            seg.end = seg.start + 1.0

        chunks = split_long_segment_text(seg.text) if split_long else [seg.text]
        total_chars = sum(max(1, len(c)) for c in chunks)

        cur_start = seg.start
        for i, chunk in enumerate(chunks):
            frac = len(chunk) / total_chars
            dur = max(0.7, (seg.end - seg.start) * frac)
            cur_end = seg.end if i == len(chunks) - 1 else min(seg.end, cur_start + dur)

            prefix = f"[{seg.speaker}] " if include_speaker else ""

            lines.append(str(idx))
            lines.append(f"{sec_to_srt_time(cur_start)} --> {sec_to_srt_time(cur_end)}")
            lines.append(prefix + chunk)
            lines.append("")

            idx += 1
            cur_start = cur_end

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_txt(segments: List[Segment], out_path: Path) -> None:
    lines = []
    last_speaker = None
    for seg in segments:
        if seg.speaker != last_speaker:
            lines.append(f"\n[{seg.speaker}]")
            last_speaker = seg.speaker
        lines.append(seg.text)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input MP4/MOV/MKV/audio file")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--language", default="ru", help="ISO language code, default ru")
    parser.add_argument("--audio_bitrate", default="32k", help="Compressed MP3 bitrate, default 32k")
    parser.add_argument("--max_upload_mb", type=float, default=MAX_UPLOAD_MB_DEFAULT)
    parser.add_argument("--chunk_seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--force_chunks", action="store_true", help="Always split audio into chunks")
    parser.add_argument("--keep_hallucination_filter", action="store_true", help="Disable removal of common subtitle-credit hallucinations")
    parser.add_argument("--no_speaker_prefix", action="store_true", help="Do not write [SPEAKER_X] before SRT text")
    parser.add_argument("--known_speaker", action="append", default=[], help="Optional: Name=path_to_2_10_sec_reference.wav. Can repeat up to 4 times.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    require_tool("ffmpeg")
    require_tool("ffprobe")
    ensure_dir(args.outdir)

    work = args.outdir / "work"
    ensure_dir(work)

    stem = args.input.stem
    audio_mp3 = work / f"{stem}.16k.{args.audio_bitrate}.mp3"

    log("[1/5] Extracting compressed mono audio...")
    extract_audio_mp3(args.input, audio_mp3, bitrate=args.audio_bitrate)
    duration = ffprobe_duration(audio_mp3)
    size_mb = file_mb(audio_mp3)
    log(f"[AUDIO] duration={duration/60:.1f} min, size={size_mb:.2f} MB")

    known_names: List[str] = []
    known_paths: List[Path] = []
    for item in args.known_speaker:
        if "=" not in item:
            raise RuntimeError(f"--known_speaker must be Name=path, got: {item}")
        name, path = item.split("=", 1)
        known_names.append(name.strip())
        known_paths.append(Path(path.strip()))

    client = OpenAI()

    all_segments: List[Segment] = []
    raw_outputs: List[Dict[str, Any]] = []

    use_chunks = args.force_chunks or size_mb > args.max_upload_mb

    if not use_chunks:
        log("[2/5] Audio is under upload limit. Sending as one file for best speaker consistency.")
        raw = call_openai_diarize(
            client=client,
            audio_path=audio_mp3,
            language=args.language,
            known_speaker_names=known_names or None,
            known_speaker_reference_paths=known_paths or None,
        )
        raw_outputs.append(raw)
        all_segments.extend(parse_segments(raw, 0, 0.0, remove_hallucinations=not args.keep_hallucination_filter))
    else:
        log("[2/5] Audio too large or --force_chunks enabled. Splitting into chunks.")
        chunks_dir = work / "chunks"
        chunks = split_audio(audio_mp3, chunks_dir, args.chunk_seconds)
        offset = 0.0
        for i, chunk in enumerate(chunks):
            chunk_dur = ffprobe_duration(chunk)
            raw = call_openai_diarize(
                client=client,
                audio_path=chunk,
                language=args.language,
                known_speaker_names=known_names or None,
                known_speaker_reference_paths=known_paths or None,
            )
            raw_outputs.append(raw)
            all_segments.extend(parse_segments(raw, i, offset, remove_hallucinations=not args.keep_hallucination_filter))
            offset += chunk_dur

    all_segments.sort(key=lambda s: (s.start, s.end))

    json_path = args.outdir / f"{stem}.ru.diarized.json"
    srt_path = args.outdir / f"{stem}.ru.diarized.srt"
    txt_path = args.outdir / f"{stem}.ru.diarized.txt"
    raw_path = args.outdir / f"{stem}.openai_raw.json"

    log("[3/5] Saving raw JSON...")
    raw_path.write_text(json.dumps(raw_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    json_path.write_text(json.dumps([asdict(s) for s in all_segments], ensure_ascii=False, indent=2), encoding="utf-8")

    log("[4/5] Writing SRT/TXT...")
    write_srt(all_segments, srt_path, include_speaker=not args.no_speaker_prefix)
    write_txt(all_segments, txt_path)

    speakers = sorted(set(s.speaker for s in all_segments))
    log("[5/5] Done.")
    log(f"Segments: {len(all_segments)}")
    log(f"Speakers: {', '.join(speakers) if speakers else 'none'}")
    log(f"SRT: {srt_path}")
    log(f"JSON: {json_path}")
    log(f"TXT: {txt_path}")

    if use_chunks:
        log("")
        log("[WARN] You used chunked mode. Speaker labels may reset between chunks.")
        log("For best diarization consistency, use one compressed MP3 under 25 MB if possible.")


if __name__ == "__main__":
    main()
