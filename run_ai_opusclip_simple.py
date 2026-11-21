#!/usr/bin/env python3
"""One-command helper: transcribe (if needed) -> ai_opusclip.py -> burn subtitles.

Example:
    python run_ai_opusclip_simple.py --video talk.mp4 --language en
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ai_opusclip import main as run_ai_opusclip
from freedom_clips.ffmpeg import require_ffmpeg, run as run_ffmpeg
from freedom_clips.llm import get_openai_client
from freedom_clips.srt import assess_srt_quality, parse_srt


# Set from the command line in main().
VIDEO_PATH: Path
LANGUAGE: str
SRT_PATH: Path
OUT_DIR: Path
WHISPER_MODEL = "whisper-1"
MIN_SRT_CUES = 3
MIN_CLIP_SEC = 10
MAX_CLIP_SEC = 90
BURN_SUBTITLES = True


def _parse_args(argv: list[str] | None = None) -> None:
    global VIDEO_PATH, LANGUAGE, SRT_PATH, OUT_DIR, BURN_SUBTITLES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path, help="Source video file")
    parser.add_argument("--language", required=True, help="Spoken language ISO-639-1 code for Whisper, e.g. en, ru, hi")
    parser.add_argument("--srt", type=Path, default=None, help="SRT path (default: next to the video)")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: <video>_clips_out)")
    parser.add_argument("--no-burn-subtitles", action="store_true", help="Skip burning subtitles into clips")
    args = parser.parse_args(argv)
    VIDEO_PATH = args.video
    LANGUAGE = args.language.strip().lower()
    SRT_PATH = args.srt or VIDEO_PATH.with_suffix(".srt")
    OUT_DIR = args.out or VIDEO_PATH.with_name(f"{VIDEO_PATH.stem}_clips_out")
    BURN_SUBTITLES = not args.no_burn_subtitles


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE lines from .env without adding a dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _validate_config() -> None:
    if not (2 <= len(LANGUAGE) <= 3 and LANGUAGE.isalpha()):
        raise ValueError(f"--language must be an ISO-639-1 code such as en, ru or hi, got {LANGUAGE!r}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    if not VIDEO_PATH.is_file():
        raise ValueError(f"VIDEO_PATH is not a file: {VIDEO_PATH}")


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    print(f"Extracting audio for Whisper: {audio_path}", flush=True)
    run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-af", "aresample=async=1:first_pts=0,highpass=f=80,lowpass=f=7600,dynaudnorm=f=150:g=15",
            "-c:a", "mp3",
            "-b:a", "96k",
            str(audio_path),
        ]
    )


def _transcribe_to_srt(audio_path: Path, srt_path: Path) -> None:
    print(f"Transcribing with OpenAI Whisper ({WHISPER_MODEL}, language={LANGUAGE})...", flush=True)
    client = get_openai_client()
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            language=LANGUAGE,
            response_format="srt",
            temperature=0,
            prompt=(
                "Transcribe the speech faithfully. Do not repeat filler words to fill silence. "
                "If a segment is silent or unintelligible, leave it out instead of guessing."
            ),
        )
    srt_text = str(transcript).strip() + "\n"
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(srt_text, encoding="utf-8")
    print(f"Saved SRT: {srt_path}", flush=True)


def _srt_summary(srt_path: Path) -> tuple[bool, str]:
    try:
        cues = parse_srt(srt_path)
    except Exception as exc:
        return False, f"failed to parse: {exc}"

    report = assess_srt_quality(cues, min_cues=MIN_SRT_CUES, min_span_sec=MIN_CLIP_SEC)
    return report.ok, report.reason


def ensure_srt() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SRT_PATH.exists():
        valid, summary = _srt_summary(SRT_PATH)
        if valid:
            print(f"Reusing existing SRT: {SRT_PATH} ({summary})", flush=True)
            return SRT_PATH
        print(f"Existing SRT is unusable and will be overwritten: {SRT_PATH} ({summary})", flush=True)

    require_ffmpeg(need_encoder=True)
    audio_path = OUT_DIR / f"{VIDEO_PATH.stem}_whisper.mp3"
    _extract_audio(VIDEO_PATH, audio_path)
    print(f"Saved Whisper MP3 for reuse/debugging: {audio_path}", flush=True)
    _transcribe_to_srt(audio_path, SRT_PATH)

    valid, summary = _srt_summary(SRT_PATH)
    if not valid:
        raise RuntimeError(f"Whisper produced an unusable SRT at {SRT_PATH}: {summary}")
    print(f"Generated usable SRT: {SRT_PATH} ({summary})", flush=True)
    return SRT_PATH


def _subtitles_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def _burn_subtitles(video_path: Path, srt_path: Path, out_path: Path) -> None:
    print(f"Burning subtitles: {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subtitles_path = _subtitles_filter_path(srt_path)
    run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{subtitles_path}'",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
    )


def burn_subtitles_for_rendered_clips(rendered_list_path: Path) -> None:
    if not rendered_list_path.exists():
        raise FileNotFoundError(f"Rendered clips list not found: {rendered_list_path}")

    burned: list[str] = []
    for raw_line in rendered_list_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        video_path = Path(raw_line.strip())
        srt_path = video_path.with_suffix(".srt")
        if not srt_path.exists():
            raise FileNotFoundError(f"Expected matching SRT for burned subtitles: {srt_path}")
        out_path = video_path.with_name(f"{video_path.stem}_subtitled{video_path.suffix}")
        _burn_subtitles(video_path, srt_path, out_path)
        burned.append(str(out_path.resolve()))

    (OUT_DIR / "rendered_files_subtitled.txt").write_text(
        "\n".join(burned) + ("\n" if burned else ""),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    _parse_args(argv)
    _load_dotenv()
    _validate_config()
    srt_path = ensure_srt()

    print("Running ai_opusclip.py with default parameters...", flush=True)
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "ai_opusclip.py",
            "--video", str(VIDEO_PATH),
            "--srt", str(srt_path),
            "--out", str(OUT_DIR),
            "--source-language", LANGUAGE,
            "--min-sec", str(MIN_CLIP_SEC),
            "--max-sec", str(MAX_CLIP_SEC),
            "--min-clips", "5", 
            "--max-clips", "12"
        ]
        run_ai_opusclip()
    finally:
        sys.argv = original_argv

    if BURN_SUBTITLES:
        burn_subtitles_for_rendered_clips(OUT_DIR / "rendered_files.txt")

    print(f"Done. Clips output directory: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
