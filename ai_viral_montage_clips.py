#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freedom_clips.ffmpeg import media_duration, render_montage_clip, require_ffmpeg
from freedom_clips.llm import DEFAULT_MODEL, call_structured_json, get_openai_client
from freedom_clips.plan_cache import PlanCache, make_plan_cache_key
from freedom_clips.prompts.montage import (
    PROMPT_VERSION,
    build_instructions,
    get_montage_schema,
    units_payload,
)
from freedom_clips.srt import merge_cues, parse_srt, write_montage_srt
from freedom_clips.text_overlay import make_text_overlay_png
from freedom_clips.validation import (
    require_existing_file,
    sanitize_filename,
    validate_montage_plan,
)


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-cost viral montage clip generator using original footage, SRT, OpenAI planning, Pillow overlays and ffmpeg."
    )
    parser.add_argument("--video", "--input-mp4", "--input_mp4", dest="video", required=True, type=Path)
    parser.add_argument("--srt", required=True, type=Path)
    parser.add_argument("--out", "--outdir", dest="out", type=Path, default=Path("montage_out"))
    parser.add_argument("--n-clips", "--n_clips", dest="n_clips", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-unit-sec", type=float, default=2.0)
    parser.add_argument("--max-unit-sec", type=float, default=12.5)
    parser.add_argument("--min-clip-sec", type=float, default=15.0)
    parser.add_argument("--max-clip-sec", type=float, default=25.0)
    parser.add_argument("--edit-mode", choices=["chronological", "montage"], default="chronological")
    parser.add_argument("--content-context", default="")
    parser.add_argument("--layout", choices=["crop", "blur"], default="crop")
    parser.add_argument("--font-path")
    parser.add_argument("--hook-font-size", type=int, default=72)
    parser.add_argument("--overlay-font-size", type=int, default=52)
    parser.add_argument("--cta-font-size", type=int, default=54)
    parser.add_argument("--cta-seconds", type=float, default=2.6)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--audio-codec", default="aac")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/plans"))
    parser.add_argument("--refresh-plan", action="store_true")
    parser.add_argument("--plan", type=Path, help="Use an existing montage plan JSON")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--verbose-ffmpeg", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_clips < 1:
        raise ValueError("--n-clips must be >= 1")
    if not (0 < args.min_unit_sec <= args.max_unit_sec):
        raise ValueError("Require 0 < --min-unit-sec <= --max-unit-sec")
    if not (0 < args.min_clip_sec <= args.max_clip_sec):
        raise ValueError("Require 0 < --min-clip-sec <= --max-clip-sec")
    if args.render_only and not args.plan:
        raise ValueError("--render-only requires --plan")
    if args.plan_only and args.render_only:
        raise ValueError("Choose only one of --plan-only or --render-only")

    require_existing_file(args.video, "video")
    require_existing_file(args.srt, "SRT")
    require_ffmpeg(need_encoder=not args.plan_only)
    args.out.mkdir(parents=True, exist_ok=True)

    cues = parse_srt(args.srt)
    if not cues:
        raise RuntimeError("No subtitle cues parsed")
    units = merge_cues(cues, min_dur=args.min_unit_sec, max_dur=args.max_unit_sec)
    if len(units) < 3:
        raise RuntimeError("Need at least 3 semantic units for montage planning")
    units_by_id = {unit.unit_id: unit for unit in units}
    _dump(args.out / "merged_units.json", units_payload(units))

    raw_plan: dict[str, Any]
    cache_hit = False
    if args.plan:
        require_existing_file(args.plan, "plan")
        raw_plan = _load(args.plan)
    else:
        instructions = build_instructions(
            n_clips=args.n_clips,
            min_clip_sec=args.min_clip_sec,
            max_clip_sec=args.max_clip_sec,
            edit_mode=args.edit_mode,
            content_context=args.content_context,
        )
        payload = units_payload(units)
        planning_input = json.dumps(payload, ensure_ascii=False, indent=2)
        cache_key = make_plan_cache_key(
            {
                "kind": "montage",
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "units": payload,
                "n_clips": args.n_clips,
                "min_clip_sec": args.min_clip_sec,
                "max_clip_sec": args.max_clip_sec,
                "edit_mode": args.edit_mode,
                "content_context": args.content_context,
            }
        )
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
                user_input="Create montage plans from these timecoded semantic units:\n\n" + planning_input,
                schema=get_montage_schema(args.n_clips),
                schema_name="montage_clip_plan",
                max_output_tokens=max(2800, args.n_clips * 800),
            )
            cache.save(cache_key, raw_plan)

    _dump(args.out / "raw_plan.json", raw_plan)
    final_plan = validate_montage_plan(
        raw_plan,
        units_by_id,
        min_clip_sec=args.min_clip_sec,
        max_clip_sec=args.max_clip_sec,
        edit_mode=args.edit_mode,
        max_clips=args.n_clips,
    )
    final_plan.update(
        {
            "version": "0.2.0",
            "source_video": str(args.video),
            "source_srt": str(args.srt),
            "edit_mode": args.edit_mode,
            "cache_hit": cache_hit,
        }
    )
    _dump(args.out / "final_montage_plan.json", final_plan)

    print(f"Validated {len(final_plan['clips'])} montage clips. cache_hit={cache_hit}")
    for clip in final_plan["clips"]:
        print(
            f"clip={clip['clip_id']} score={clip['viral_score']} "
            f"duration={clip['duration_sec']:.2f}s | {clip['title']}"
        )

    if args.plan_only:
        print("Plan-only mode: no video rendered.")
        return

    _ = media_duration(args.video)  # early ffprobe failure before rendering any clip
    outputs: list[str] = []
    for sequence, clip in enumerate(final_plan["clips"], start=1):
        title = sanitize_filename(clip["title"], 65)
        clip_dir = args.out / f"{sequence:02d}_{clip['viral_score']:03d}_{title}"
        assets_dir = clip_dir / "overlays"
        assets_dir.mkdir(parents=True, exist_ok=True)

        hook_png = make_text_overlay_png(
            clip["hook"]["text"],
            assets_dir / "hook.png",
            y_ratio=0.075,
            font_size=args.hook_font_size,
            font_path=args.font_path,
            max_lines=3,
        )

        overlay_windows: list[tuple[Path, float, float]] = []
        cursor = 0.0
        for beat_index, beat in enumerate(clip["beats"], start=1):
            unit = units_by_id[int(beat["unit_id"])]
            beat_duration = unit.duration / float(beat["speed"])
            text = str(beat.get("overlay") or "").strip()
            if text:
                path = make_text_overlay_png(
                    text,
                    assets_dir / f"beat_{beat_index:02d}.png",
                    y_ratio=0.79,
                    font_size=args.overlay_font_size,
                    font_path=args.font_path,
                    max_lines=2,
                )
                overlay_windows.append((path, cursor, cursor + beat_duration))
            cursor += beat_duration

        cta_text = str(clip.get("cta_question") or "").strip()
        cta_png = None
        if cta_text:
            cta_png = make_text_overlay_png(
                cta_text,
                assets_dir / "cta.png",
                y_ratio=0.10,
                font_size=args.cta_font_size,
                font_path=args.font_path,
                max_lines=3,
            )

        out_mp4 = clip_dir / f"{sequence:02d}_{title}.mp4"
        render_montage_clip(
            video=args.video,
            out_path=out_mp4,
            beats=clip["beats"],
            units_by_id=units_by_id,
            hook_png=hook_png,
            hook_seconds=float(clip["hook"]["display_seconds"]),
            overlay_windows=overlay_windows,
            cta_png=cta_png,
            cta_seconds=args.cta_seconds,
            layout=args.layout,
            video_codec=args.video_codec,
            audio_codec=args.audio_codec,
            crf=args.crf,
            preset=args.preset,
            verbose=args.verbose_ffmpeg,
        )
        write_montage_srt(clip["beats"], units_by_id, out_mp4.with_suffix(".srt"))
        _dump(clip_dir / "metadata.json", clip)
        outputs.append(str(out_mp4.resolve()))

    manifest = {
        "version": "0.2.0",
        "source_video": str(args.video.resolve()),
        "source_srt": str(args.srt.resolve()),
        "outputs": outputs,
        "theme": final_plan.get("global_theme", ""),
    }
    _dump(args.out / "manifest.json", manifest)
    print(f"Rendered {len(outputs)} montage clips into {args.out}")


if __name__ == "__main__":
    main()
