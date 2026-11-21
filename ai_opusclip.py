#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freedom_clips.ffmpeg import media_duration, render_contiguous_clip, require_ffmpeg
from freedom_clips.llm import DEFAULT_MODEL, call_structured_json, get_openai_client
from freedom_clips.plan_cache import PlanCache, make_plan_cache_key
from freedom_clips.prompts.opusclip import PROMPT_VERSION, build_instructions, get_clip_schema
from freedom_clips.srt import build_timecoded_transcript, parse_srt, write_clip_srt
from freedom_clips.text_overlay import make_text_overlay_png
from freedom_clips.validation import require_existing_file, sanitize_filename, validate_contiguous_clips


def _json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, clips: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "start_sec", "end_sec", "duration_sec", "viral_score",
        "title", "hook_text", "caption", "why_selected",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip in clips:
            writer.writerow({field: clip.get(field) for field in fields})


def _load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topic- and language-agnostic OpusClip-like vertical clip generator using SRT + OpenAI + ffmpeg."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--srt", required=True, type=Path, help="Primary transcript/subtitle SRT")
    parser.add_argument("--secondary-srt", type=Path, help="Optional second-language/context SRT")
    parser.add_argument("--out", type=Path, default=Path("clips_out"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-clips", type=int, default=5)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--min-sec", type=int, default=30)
    parser.add_argument("--max-sec", type=int, default=90)
    parser.add_argument("--max-overlap-sec", type=float, default=0.0)
    parser.add_argument("--source-language", default="auto")
    parser.add_argument("--hook-language", default="same")
    parser.add_argument("--editorial-style", default="auto")
    parser.add_argument("--content-context", default="")
    parser.add_argument("--layout", choices=["crop", "blur"], default="crop")
    parser.add_argument("--hook-seconds", type=float, default=4.0)
    parser.add_argument("--font-path")
    parser.add_argument("--font-size", type=int, default=72)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--audio-codec", default="aac")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/plans"))
    parser.add_argument("--refresh-plan", action="store_true")
    parser.add_argument("--plan", type=Path, help="Use an existing plan JSON instead of calling the LLM")
    parser.add_argument("--plan-only", action="store_true", help="Create/validate the plan but do not render video")
    parser.add_argument("--render-only", action="store_true", help="Render from --plan; never call the LLM")
    parser.add_argument("--no-secondary-in-prompt", action="store_true")
    parser.add_argument("--verbose-ffmpeg", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_clips < 1 or args.max_clips < args.min_clips:
        raise ValueError("Require 1 <= --min-clips <= --max-clips")
    if args.min_sec <= 0 or args.max_sec < args.min_sec:
        raise ValueError("Require 0 < --min-sec <= --max-sec")
    if args.render_only and not args.plan:
        raise ValueError("--render-only requires --plan")
    if args.plan_only and args.render_only:
        raise ValueError("Choose only one of --plan-only or --render-only")

    require_existing_file(args.video, "video")
    require_existing_file(args.srt, "SRT")
    if args.secondary_srt:
        require_existing_file(args.secondary_srt, "secondary SRT")

    require_ffmpeg(need_encoder=not args.plan_only)
    args.out.mkdir(parents=True, exist_ok=True)

    primary_cues = parse_srt(args.srt)
    if not primary_cues:
        raise RuntimeError("No subtitle cues parsed from primary SRT")
    secondary_cues = parse_srt(args.secondary_srt) if args.secondary_srt else None
    transcript = build_timecoded_transcript(
        primary_cues,
        secondary_cues,
        include_secondary=bool(secondary_cues) and not args.no_secondary_in_prompt,
    )
    (args.out / "transcript_for_planning.txt").write_text(transcript, encoding="utf-8")
    video_duration = media_duration(args.video)

    raw_plan: dict[str, Any]
    cache_hit = False
    if args.plan:
        require_existing_file(args.plan, "plan")
        raw_plan = _load_plan(args.plan)
    else:
        instructions = build_instructions(
            min_clips=args.min_clips,
            max_clips=args.max_clips,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            source_language=args.source_language,
            hook_language=args.hook_language,
            editorial_style=args.editorial_style,
            content_context=args.content_context,
            max_overlap_sec=args.max_overlap_sec,
        )
        cache_payload = {
            "kind": "opusclip",
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "transcript": transcript,
            "min_clips": args.min_clips,
            "max_clips": args.max_clips,
            "min_sec": args.min_sec,
            "max_sec": args.max_sec,
            "max_overlap_sec": args.max_overlap_sec,
            "source_language": args.source_language,
            "hook_language": args.hook_language,
            "editorial_style": args.editorial_style,
            "content_context": args.content_context,
        }
        cache_key = make_plan_cache_key(cache_payload)
        cache = PlanCache(args.cache_dir)
        raw_plan = None  # type: ignore[assignment]
        if not args.refresh_plan:
            raw_plan = cache.load(cache_key)  # type: ignore[assignment]
            cache_hit = raw_plan is not None
        if raw_plan is None:
            client = get_openai_client()
            raw_plan = call_structured_json(
                client=client,
                model=args.model,
                instructions=instructions,
                user_input="Return the clip plan for this timecoded transcript:\n\n" + transcript,
                schema=get_clip_schema(args.min_clips, args.max_clips),
                schema_name="continuous_clip_plan",
                max_output_tokens=max(2500, args.max_clips * 420),
            )
            cache.save(cache_key, raw_plan)

    _json_dump(args.out / "raw_plan.json", raw_plan)
    raw_clips = raw_plan.get("clips")
    if not isinstance(raw_clips, list):
        raise RuntimeError("Plan does not contain a clips array")

    selected = validate_contiguous_clips(
        raw_clips,
        primary_cues,
        video_duration,
        args.min_sec,
        args.max_sec,
        args.max_clips,
        args.max_overlap_sec,
    )
    if not selected:
        raise RuntimeError("No valid clips remained after timestamp/duration/overlap validation")

    final_plan = {
        "version": "0.2.0",
        "source_video": str(args.video),
        "source_srt": str(args.srt),
        "analysis": raw_plan.get("analysis", {}),
        "cache_hit": cache_hit,
        "clips": selected,
    }
    _json_dump(args.out / "final_clip_plan.json", final_plan)
    _write_csv(args.out / "final_clip_plan.csv", selected)

    print(f"Selected {len(selected)} clips. cache_hit={cache_hit}")
    for clip in selected:
        print(
            f"{clip['rank']:02d}. score={clip['viral_score']:3d} "
            f"{clip['start_sec']:.3f}->{clip['end_sec']:.3f} | {clip['title']}"
        )

    if args.plan_only:
        print("Plan-only mode: no video rendered.")
        return

    rendered: list[str] = []
    for clip in selected:
        rank = int(clip["rank"])
        title = sanitize_filename(clip["title"], 70)
        clip_dir = args.out / f"{rank:02d}_{clip['viral_score']:03d}_{title}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        hook_png = make_text_overlay_png(
            clip["hook_text"],
            clip_dir / "hook.png",
            y_ratio=0.08,
            font_size=args.font_size,
            font_path=args.font_path,
            max_lines=3,
        )
        out_mp4 = clip_dir / f"{rank:02d}_{title}.mp4"
        render_contiguous_clip(
            video=args.video,
            out_path=out_mp4,
            start=float(clip["start_sec"]),
            end=float(clip["end_sec"]),
            hook_png=hook_png,
            hook_seconds=args.hook_seconds,
            layout=args.layout,
            video_codec=args.video_codec,
            audio_codec=args.audio_codec,
            crf=args.crf,
            preset=args.preset,
            verbose=args.verbose_ffmpeg,
        )
        write_clip_srt(
            primary_cues,
            float(clip["start_sec"]),
            float(clip["end_sec"]),
            out_mp4.with_suffix(".srt"),
        )
        _json_dump(clip_dir / "metadata.json", clip)
        rendered.append(str(out_mp4.resolve()))

    (args.out / "rendered_files.txt").write_text("\n".join(rendered) + "\n", encoding="utf-8")
    print(f"Rendered {len(rendered)} clips into {args.out}")


if __name__ == "__main__":
    main()
