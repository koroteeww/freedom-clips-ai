import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


def parse_srt_timestamp(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    m = re.match(r"(\d+):(\d+):(\d+(?:\.\d+)?)", ts)
    if not m:
        raise ValueError(f"Bad SRT timestamp: {ts}")
    h, mnt, sec = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(sec)


def seconds_to_srt_time(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_ass_time(seconds: float) -> str:
    cs = int(round((seconds - int(seconds)) * 100))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean_srt_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues: list[Cue] = []

    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        if len(lines) < 2:
            continue

        try:
            index = int(lines[0])
            time_line = lines[1]
            text_lines = lines[2:]
        except ValueError:
            index = len(cues) + 1
            time_line = lines[0]
            text_lines = lines[1:]

        if "-->" not in time_line:
            continue

        start_raw, end_raw = [x.strip() for x in time_line.split("-->", 1)]
        start = parse_srt_timestamp(start_raw)
        end = parse_srt_timestamp(end_raw.split()[0])
        text = clean_srt_text(" ".join(text_lines))

        if text:
            cues.append(Cue(index=index, start=start, end=end, text=text))

    return cues


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ffprobe_duration(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s\-а-яА-ЯёЁ\u0900-\u097F]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len] or "clip"


def build_timecoded_transcript(en_cues: list[Cue], hi_cues: list[Cue] | None, include_hi: bool) -> str:
    lines = []
    for i, en in enumerate(en_cues):
        start = seconds_to_srt_time(en.start)
        end = seconds_to_srt_time(en.end)
        if include_hi and hi_cues and i < len(hi_cues) and abs(hi_cues[i].start - en.start) < 5:
            lines.append(f"[{start} --> {end}] EN: {en.text}\nHI: {hi_cues[i].text}")
        else:
            lines.append(f"[{start} --> {end}] EN: {en.text}")
    return "\n".join(lines)


def get_clip_schema(min_clips: int, max_clips: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "clips": {
                "type": "array",
                "minItems": min_clips,
                "maxItems": max_clips,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rank": {"type": "integer"},
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "viral_score": {"type": "integer"},
                        "title_en": {"type": "string"},
                        "hook_hi": {"type": "string"},
                        "caption_en": {"type": "string"},
                        "why_viral": {"type": "string"},
                    },
                    "required": [
                        "rank", "start_sec", "end_sec", "viral_score",
                        "title_en", "hook_hi", "caption_en", "why_viral"
                    ],
                },
            }
        },
        "required": ["clips"],
    }


def call_openai_for_clip_plan(
    transcript: str,
    model: str,
    min_clips: int,
    max_clips: int,
    min_sec: int,
    max_sec: int,
) -> dict[str, Any]:
    client = OpenAI()
    instructions = f"""
You are an expert short-form video editor like OpusClip.

Task:
Analyze a long Hindi speech about digital freedom.
Select viral vertical short clips for TikTok, Instagram Reels, and YouTube Shorts.

Editorial style:
- Motivational, visionary, guru-like, serious.
- Theme: digital freedom, privacy, censorship, personal sovereignty, technology, courage.
- Prefer clips that feel like standalone truth bombs.
- Avoid boring introductions, greetings, housekeeping, repeated ideas.
- Avoid clips that require too much missing context.

Clip rules:
- Return between {min_clips} and {max_clips} clips.
- Each clip must be between {min_sec} and {max_sec} seconds.
- Clips should be contiguous ranges from the original video.
- Start and end should be near natural sentence/thought boundaries.
- Avoid overlapping clips.
- Use original timestamps.
- hook_hi must be in Hindi / Devanagari.
- hook_hi should be short, punchy, max 12 words.
- Do not invent facts not supported by the transcript.
"""
    schema = get_clip_schema(min_clips, max_clips)
    user_prompt = f"Return JSON only.\n\nTimecoded transcript:\n{transcript}"

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "viral_clip_plan",
                    "strict": True,
                    "schema": schema,
                }
            },
            max_output_tokens=12000,
        )
        return json.loads(response.output_text)
    except Exception as e:
        print("\nStructured output failed, trying plain JSON fallback:", repr(e))
        response = client.responses.create(
            model=model,
            input=instructions + "\nReturn valid JSON only matching this schema:\n"
            + json.dumps(schema, ensure_ascii=False) + "\n\n" + user_prompt,
            max_output_tokens=12000,
        )
        text = response.output_text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


def snap_to_nearest_cue_start(sec: float, cues: list[Cue]) -> float:
    return min(cues, key=lambda c: abs(c.start - sec)).start


def snap_to_nearest_cue_end(sec: float, cues: list[Cue]) -> float:
    return min(cues, key=lambda c: abs(c.end - sec)).end


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float, tolerance: float = 3.0) -> bool:
    return max(a_start, b_start) < min(a_end, b_end) - tolerance


def validate_and_select_clips(
    raw_clips: list[dict[str, Any]],
    en_cues: list[Cue],
    video_duration: float,
    min_clips: int,
    max_clips: int,
    min_sec: int,
    max_sec: int,
) -> list[dict[str, Any]]:
    normalized = []

    for c in raw_clips:
        start = snap_to_nearest_cue_start(float(c["start_sec"]), en_cues)
        end = snap_to_nearest_cue_end(float(c["end_sec"]), en_cues)
        start = max(0, start)
        end = min(video_duration, end)
        duration = end - start

        if duration < min_sec:
            end = min(video_duration, start + min_sec)
            duration = end - start
        if duration > max_sec:
            end = start + max_sec
            duration = end - start
        if duration < min_sec or duration > max_sec or end <= start:
            continue

        item = dict(c)
        item["start_sec"] = round(start, 3)
        item["end_sec"] = round(end, 3)
        item["duration_sec"] = round(duration, 3)
        item["viral_score"] = int(c.get("viral_score", 0))
        normalized.append(item)

    normalized.sort(key=lambda x: x["viral_score"], reverse=True)
    selected = []
    for c in normalized:
        if len(selected) >= max_clips:
            break
        if any(overlaps(c["start_sec"], c["end_sec"], s["start_sec"], s["end_sec"]) for s in selected):
            continue
        selected.append(c)

    if len(selected) < min_clips:
        print(f"\nWARNING: only selected {len(selected)} non-overlapping valid clips.")
    return selected


def escape_ass_text(text: str) -> str:
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    wrapped = textwrap.wrap(text, width=22)
    return r"\N".join(wrapped[:3])


def write_hook_ass(ass_path: Path, hook_text: str, hook_seconds: float, font_name: str, font_size: int) -> None:
    hook = escape_ass_text(hook_text)
    content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,5,2,8,80,80,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,{seconds_to_ass_time(0)},{seconds_to_ass_time(hook_seconds)},Hook,,0,0,0,,{{\\an8\\pos(540,170)}}{hook}
"""
    ass_path.write_text(content, encoding="utf-8-sig")


def make_filter_complex(layout: str) -> str:
    if layout == "crop":
        return "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,subtitles=hook.ass[v]"
    if layout == "blur":
        return (
            "[0:v]split=2[fg][bg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=28[bgv];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fgv];"
            "[bgv][fgv]overlay=(W-w)/2:(H-h)/2,subtitles=hook.ass[v]"
        )
    raise ValueError(f"Unknown layout: {layout}")


def render_clip(
    video: Path,
    clip: dict[str, Any],
    out_dir: Path,
    idx: int,
    layout: str,
    hook_seconds: float,
    font_name: str,
    font_size: int,
    crf: int,
    preset: str,
) -> Path:
    title = sanitize_filename(clip.get("title_en", f"clip_{idx}"))
    score = int(clip.get("viral_score", 0))
    clip_dir = out_dir / f"{idx:02d}_{score}_{title}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    write_hook_ass(clip_dir / "hook.ass", clip.get("hook_hi", ""), hook_seconds, font_name, font_size)
    out_mp4 = clip_dir / f"{idx:02d}_{score}_{title}.mp4"
    start = float(clip["start_sec"])
    duration = float(clip["end_sec"]) - float(clip["start_sec"])

    run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video.resolve()),
        "-t", f"{duration:.3f}",
        "-filter_complex", make_filter_complex(layout),
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(out_mp4.resolve()),
    ], cwd=clip_dir)

    (clip_dir / "metadata.json").write_text(json.dumps(clip, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_mp4


def write_csv(plan_path: Path, clips: list[dict[str, Any]]) -> None:
    with plan_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["rank", "start_sec", "end_sec", "duration_sec", "viral_score", "title_en", "hook_hi", "caption_en", "why_viral"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in clips:
            writer.writerow({k: c.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="AI OpusClip-like vertical clip generator using ffmpeg + OpenAI API.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--en", required=True)
    parser.add_argument("--hi", required=False)
    parser.add_argument("--out", default="ai_clips_out")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--min-clips", type=int, default=10)
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--min-sec", type=int, default=30)
    parser.add_argument("--max-sec", type=int, default=90)
    parser.add_argument("--layout", choices=["crop", "blur"], default="crop")
    parser.add_argument("--hook-seconds", type=float, default=5.0)
    parser.add_argument("--font-name", default="Nirmala UI")
    parser.add_argument("--font-size", type=int, default=76)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-hi-in-prompt", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    en_srt = Path(args.en)
    hi_srt = Path(args.hi) if args.hi else None
    out_dir = Path(args.out)

    if not video.exists(): raise FileNotFoundError(video)
    if not en_srt.exists(): raise FileNotFoundError(en_srt)
    if hi_srt and not hi_srt.exists(): raise FileNotFoundError(hi_srt)
    if shutil.which("ffmpeg") is None: raise RuntimeError("ffmpeg not found in PATH.")
    if shutil.which("ffprobe") is None: raise RuntimeError("ffprobe not found in PATH.")
    if not os.environ.get("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    out_dir.mkdir(parents=True, exist_ok=True)
    en_cues = parse_srt(en_srt)
    hi_cues = parse_srt(hi_srt) if hi_srt else None
    video_duration = ffprobe_duration(video)
    transcript = build_timecoded_transcript(en_cues, hi_cues, include_hi=bool(hi_cues) and not args.no_hi_in_prompt)
    (out_dir / "transcript_for_gpt.txt").write_text(transcript, encoding="utf-8")

    raw_plan = call_openai_for_clip_plan(transcript, args.model, args.min_clips, args.max_clips, args.min_sec, args.max_sec)
    (out_dir / "raw_openai_plan.json").write_text(json.dumps(raw_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = validate_and_select_clips(raw_plan["clips"], en_cues, video_duration, args.min_clips, args.max_clips, args.min_sec, args.max_sec)

    final_plan = {"source_video": str(video), "layout": args.layout, "clips": selected}
    (out_dir / "final_clip_plan.json").write_text(json.dumps(final_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "final_clip_plan.csv", selected)

    for i, c in enumerate(selected, start=1):
        print(f"{i:02d}. score={c['viral_score']} {seconds_to_srt_time(c['start_sec'])} -> {seconds_to_srt_time(c['end_sec'])} | {c['title_en']} | {c['hook_hi']}")

    if args.dry_run:
        print("\nDry run enabled. Not rendering videos.")
        return

    rendered = []
    for i, clip in enumerate(selected, start=1):
        rendered.append(str(render_clip(video, clip, out_dir, i, args.layout, args.hook_seconds, args.font_name, args.font_size, args.crf, args.preset)))
    (out_dir / "rendered_files.txt").write_text("\n".join(rendered), encoding="utf-8")
    print("\nDONE.")


if __name__ == "__main__":
    main()
