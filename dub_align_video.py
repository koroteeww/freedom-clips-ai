#!/usr/bin/env python3
"""Put a translated dub onto a video with drift-free timing.

Each dubbed line starts where the original line started. When the dub is
longer than the time available (up to the next line), the difference is
absorbed by speeding the dub up and/or slowing that video slice down, within
the limits you set. Nothing calls a paid API; everything runs in ffmpeg.

Examples:
  # tts_manifest.json from tretyakov_translate_tts_kdenlive.py
  python dub_align_video.py --video talk.mp4 --manifest talk_dub/tts_manifest.json

  # one audio file per SRT cue, named 0001_*.mp3, 0002_*.mp3, ...
  python dub_align_video.py --video talk.mp4 --srt talk.srt --dub-dir tts/ --text-srt talk.en.srt

  # keep the original sound quietly under the dub (time-warped with the video)
  python dub_align_video.py --video talk.mp4 --manifest m.json --background original --background-volume 0.15

  # inspect decisions without rendering
  python dub_align_video.py --video talk.mp4 --manifest m.json --plan-only
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from freedom_clips.dub_align import (
    MODES,
    lines_from_manifest,
    lines_from_srt_and_dir,
    plan_alignment,
    render_alignment,
)
from freedom_clips.ffmpeg import has_audio, media_duration, require_ffmpeg
from freedom_clips.srt import parse_srt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path, help="Source video")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="JSON list of {start, end, audio|raw_mp3|timed_mp3, text}")
    src.add_argument("--srt", type=Path, help="Original-language SRT with the source timing (use with --dub-dir)")
    p.add_argument("--dub-dir", type=Path, help="Folder of per-cue dub files named by 1-based cue number")
    p.add_argument("--text-srt", type=Path, help="Translated SRT whose text goes into the aligned subtitles")
    p.add_argument("--out", type=Path, help="Output MP4 (default: <video>_dubbed.mp4)")
    p.add_argument("--mode", choices=MODES, default="both",
                   help="both: split the difference between dub speed-up and video slow-down (default); "
                        "audio: only speed up the dub; video: only slow the video; none: trim overflow")
    p.add_argument("--max-audio-speed", type=float, default=1.35, help="Max dub speed-up factor (default 1.35)")
    p.add_argument("--max-video-slowdown", type=float, default=1.6, help="Max video slow-down factor (default 1.6)")
    p.add_argument("--no-use-gaps", action="store_true",
                   help="Do not let a dub run into the pause before the next line")
    p.add_argument("--background", default="none",
                   help="'none', 'original' (the video's own audio) or a path, e.g. a separated music track")
    p.add_argument("--background-volume", type=float, default=0.2)
    p.add_argument("--video-codec", default="libx264")
    p.add_argument("--audio-codec", default="aac")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--jobs", type=int, default=2, help="Parallel ffmpeg jobs for video slices")
    p.add_argument("--plan-only", action="store_true", help="Write the alignment plan JSON and stop")
    p.add_argument("--keep-work", action="store_true", help="Keep intermediate slices for debugging")
    p.add_argument("--verbose-ffmpeg", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ffmpeg(need_encoder=not args.plan_only)
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    if args.manifest:
        lines = lines_from_manifest(args.manifest)
    else:
        if not args.dub_dir or not args.dub_dir.is_dir():
            raise SystemExit("--srt needs --dub-dir pointing at a folder of per-cue audio files")
        cues = parse_srt(args.srt)
        texts = [c.text for c in parse_srt(args.text_srt)] if args.text_srt else None
        lines = lines_from_srt_and_dir(cues, args.dub_dir, texts)
    if not lines:
        raise SystemExit("No dub lines found")

    if args.text_srt and args.manifest:
        texts = [c.text for c in parse_srt(args.text_srt)]
        by_index = dict(enumerate(texts, 1))
        lines = [line if line.text else replace(line, text=by_index.get(line.index, "")) for line in lines]

    plan = plan_alignment(
        lines,
        media_duration(args.video),
        mode=args.mode,
        max_audio_tempo=args.max_audio_speed,
        max_video_slowdown=args.max_video_slowdown,
        use_gaps=not args.no_use_gaps,
    )
    out = args.out or args.video.with_name(f"{args.video.stem}_dubbed.mp4")
    print(json.dumps(plan.summary(), indent=2), flush=True)
    for warning in plan.warnings[:20]:
        print("WARN", warning, flush=True)
    if len(plan.warnings) > 20:
        print(f"WARN ... and {len(plan.warnings) - 20} more (see the plan JSON)", flush=True)

    if args.plan_only:
        plan_path = out.with_suffix(".alignment.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Plan: {plan_path}")
        return 0

    background: Path | None
    if args.background == "none":
        background = None
    elif args.background == "original":
        if not has_audio(args.video):
            raise SystemExit("--background original: the video has no audio stream")
        background = args.video
    else:
        background = Path(args.background)
        if not background.is_file():
            raise SystemExit(f"Background audio not found: {background}")

    result = render_alignment(
        plan,
        video=args.video,
        out_path=out,
        work_dir=out.with_name(f"{out.stem}_align_work"),
        background=background,
        background_volume=args.background_volume,
        video_codec=args.video_codec,
        audio_codec=args.audio_codec,
        crf=args.crf,
        preset=args.preset,
        jobs=args.jobs,
        keep_work=args.keep_work,
        verbose=args.verbose_ffmpeg,
    )
    print(f"Video:     {result.output} ({result.output_duration:.2f}s)")
    print(f"Subtitles: {result.aligned_srt}")
    print(f"Report:    {result.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
