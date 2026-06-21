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
        raise ValueError(f"Bad timestamp: {ts}")
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
    cues = []

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


def build_timecoded_transcript(en_cues: list[Cue], hi_cues: list[Cue] | None) -> str:
    lines = []
    for i, en in enumerate(en_cues):
        start = seconds_to_srt_time(en.start)
        end = seconds_to_srt_time(en.end)
        if hi_cues and i < len(hi_cues) and abs(hi_cues[i].start - en.start) < 5:
            lines.append(f"[{start} --> {end}] EN: {en.text}\nHI: {hi_cues[i].text}")
        else:
            lines.append(f"[{start} --> {end}] EN: {en.text}")
    return "\n".join(lines)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ffprobe_duration(video: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", str(video)
    ], text=True).strip()
    return float(out)


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s\-а-яА-ЯёЁ\u0900-\u097F]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len] or "montage"


def montage_schema(max_montages: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "montages": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_montages,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rank": {"type": "integer"},
                        "viral_score": {"type": "integer"},
                        "title_en": {"type": "string"},
                        "hook_hi": {"type": "string"},
                        "caption_en": {"type": "string"},
                        "why_viral": {"type": "string"},
                        "segments": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start_sec": {"type": "number"},
                                    "end_sec": {"type": "number"},
                                    "segment_role": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["start_sec", "end_sec", "segment_role", "reason"],
                            },
                        },
                    },
                    "required": ["rank", "viral_score", "title_en", "hook_hi", "caption_en", "why_viral", "segments"],
                },
            }
        },
        "required": ["montages"],
    }


def call_openai_for_montages(transcript: str, model: str, max_montages: int, min_seg_sec: int, max_seg_sec: int) -> dict[str, Any]:
    client = OpenAI()
    instructions = f"""
You are an expert short-form editor like OpusClip, but your task is to create montage clips.

Source:
- Hindi speech about digital freedom.
- The video may already have hardcoded Hindi subtitles.
- English and Hindi SRT text are provided for analysis.

Goal:
Create up to {max_montages} viral montage clips for TikTok, Instagram Reels, and YouTube Shorts.

Montage definition:
- Each montage is one final vertical video made from different short moments of the original speech.
- Each segment must be {min_seg_sec}-{max_seg_sec} seconds.
- Each montage must have 4-8 segments.
- Segments should come from different parts of the speech when possible.
- Each montage should feel like a coherent argument, not random quotes.

Editorial arcs:
1. Problem -> truth -> consequence -> call to courage.
2. Control -> freedom -> privacy -> responsibility.
3. Comfortable lie -> uncomfortable truth -> action.
4. One powerful idea repeated from several angles.

Rules:
- Use exact original timestamps.
- Do not invent content.
- hook_hi must be in Hindi / Devanagari.
- hook_hi max 12 words.
- caption_en is upload caption.
- title_en is filename/title.
- viral_score is 1-100.
- Final total montage duration should usually be 30-70 seconds.
"""
    schema = montage_schema(max_montages)
    user_prompt = f"Analyze this timecoded transcript and return montage clip plans.\n\nTranscript:\n{transcript}"

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_prompt,
            text={"format": {"type": "json_schema", "name": "viral_montage_plan", "strict": True, "schema": schema}},
            max_output_tokens=12000,
        )
        return json.loads(response.output_text)
    except Exception as e:
        print("\nStructured output failed, using plain JSON fallback:", repr(e))
        response = client.responses.create(
            model=model,
            input=instructions + "\n\nReturn valid JSON only matching this schema:\n"
            + json.dumps(schema, ensure_ascii=False) + "\n\n" + user_prompt,
            max_output_tokens=12000,
        )
        text = re.sub(r"^```json\s*", "", response.output_text.strip())
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


def snap_to_nearest_cue_start(sec: float, cues: list[Cue]) -> float:
    return min(cues, key=lambda c: abs(c.start - sec)).start


def snap_to_nearest_cue_end(sec: float, cues: list[Cue]) -> float:
    return min(cues, key=lambda c: abs(c.end - sec)).end


def normalize_montages(raw_montages: list[dict[str, Any]], en_cues: list[Cue], video_duration: float, max_montages: int, min_seg_sec: int, max_seg_sec: int) -> list[dict[str, Any]]:
    normalized = []
    for m in raw_montages:
        segments = []
        for s in m.get("segments", []):
            start = snap_to_nearest_cue_start(float(s["start_sec"]), en_cues)
            end = snap_to_nearest_cue_end(float(s["end_sec"]), en_cues)
            start = max(0, min(start, video_duration))
            end = max(0, min(end, video_duration))
            dur = end - start
            if dur < min_seg_sec:
                end = min(video_duration, start + min_seg_sec)
                dur = end - start
            if dur > max_seg_sec:
                end = start + max_seg_sec
                dur = end - start
            if dur < min_seg_sec or dur > max_seg_sec or end <= start:
                continue
            item = dict(s)
            item["start_sec"] = round(start, 3)
            item["end_sec"] = round(end, 3)
            item["duration_sec"] = round(dur, 3)
            segments.append(item)
        if len(segments) < 4:
            continue
        item = dict(m)
        item["segments"] = segments[:8]
        item["total_duration_sec"] = round(sum(x["duration_sec"] for x in item["segments"]), 3)
        normalized.append(item)
    normalized.sort(key=lambda x: int(x.get("viral_score", 0)), reverse=True)
    return normalized[:max_montages]


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


def base_video_filter(layout: str, add_hook: bool) -> str:
    if layout == "crop":
        chain = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
        if add_hook:
            chain += ",subtitles=hook.ass"
        return f"[0:v]{chain}[v]"
    if layout == "blur":
        suffix = ",subtitles=hook.ass" if add_hook else ""
        return (
            "[0:v]split=2[fg][bg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=28[bgv];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fgv];"
            f"[bgv][fgv]overlay=(W-w)/2:(H-h)/2{suffix}[v]"
        )
    raise ValueError(f"Unknown layout: {layout}")


def render_segment(video: Path, segment: dict[str, Any], segment_mp4: Path, work_dir: Path, layout: str, add_hook: bool, hook_text: str, hook_seconds: float, font_name: str, font_size: int, crf: int, preset: str) -> None:
    if add_hook:
        write_hook_ass(work_dir / "hook.ass", hook_text, hook_seconds, font_name, font_size)

    start = float(segment["start_sec"])
    duration = float(segment["end_sec"]) - float(segment["start_sec"])
    run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video.resolve()),
        "-t", f"{duration:.3f}",
        "-filter_complex", base_video_filter(layout, add_hook),
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "160k",
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        "-shortest",
        str(segment_mp4.resolve()),
    ], cwd=work_dir)


def concat_segments(segment_files: list[Path], output_mp4: Path, work_dir: Path) -> None:
    list_path = work_dir / "concat_list.txt"
    list_path.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in segment_files), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path.resolve()), "-c", "copy", str(output_mp4.resolve())], cwd=work_dir)


def render_montage(video: Path, montage: dict[str, Any], out_dir: Path, idx: int, layout: str, hook_seconds: float, font_name: str, font_size: int, crf: int, preset: str) -> Path:
    title = sanitize_filename(montage.get("title_en", f"montage_{idx}"))
    score = int(montage.get("viral_score", 0))
    montage_dir = out_dir / f"montage_{idx:02d}_{score}_{title}"
    montage_dir.mkdir(parents=True, exist_ok=True)

    segment_files = []
    for j, segment in enumerate(montage["segments"], start=1):
        seg_path = montage_dir / f"seg_{j:02d}.mp4"
        render_segment(video, segment, seg_path, montage_dir, layout, j == 1, montage.get("hook_hi", ""), hook_seconds, font_name, font_size, crf, preset)
        segment_files.append(seg_path)

    output_mp4 = montage_dir / f"montage_{idx:02d}_{score}_{title}.mp4"
    concat_segments(segment_files, output_mp4, montage_dir)
    (montage_dir / "metadata.json").write_text(json.dumps(montage, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_mp4


def write_csv(path: Path, montages: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "viral_score", "title_en", "hook_hi", "total_duration_sec", "caption_en", "why_viral", "segments"])
        for m in montages:
            segs = " | ".join(f"{seconds_to_srt_time(s['start_sec'])}-{seconds_to_srt_time(s['end_sec'])}: {s.get('segment_role', '')}" for s in m["segments"])
            writer.writerow([m.get("rank"), m.get("viral_score"), m.get("title_en"), m.get("hook_hi"), m.get("total_duration_sec"), m.get("caption_en"), m.get("why_viral"), segs])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OpusClip-style montage videos from different speech moments using OpenAI + ffmpeg.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--en", required=True)
    parser.add_argument("--hi", required=False)
    parser.add_argument("--out", default="montage_clips_out")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-montages", type=int, default=5)
    parser.add_argument("--min-seg-sec", type=int, default=5)
    parser.add_argument("--max-seg-sec", type=int, default=10)
    parser.add_argument("--layout", choices=["crop", "blur"], default="crop")
    parser.add_argument("--hook-seconds", type=float, default=5.0)
    parser.add_argument("--font-name", default="Nirmala UI")
    parser.add_argument("--font-size", type=int, default=76)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    en_srt = Path(args.en)
    hi_srt = Path(args.hi) if args.hi else None
    out_dir = Path(args.out)

    if not video.exists(): raise FileNotFoundError(video)
    if not en_srt.exists(): raise FileNotFoundError(en_srt)
    if hi_srt and not hi_srt.exists(): raise FileNotFoundError(hi_srt)
    if not os.environ.get("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY is not set.")
    if shutil.which("ffmpeg") is None: raise RuntimeError("ffmpeg not found in PATH.")
    if shutil.which("ffprobe") is None: raise RuntimeError("ffprobe not found in PATH.")

    out_dir.mkdir(parents=True, exist_ok=True)
    en_cues = parse_srt(en_srt)
    hi_cues = parse_srt(hi_srt) if hi_srt else None
    duration = ffprobe_duration(video)

    transcript = build_timecoded_transcript(en_cues, hi_cues)
    (out_dir / "transcript_for_gpt.txt").write_text(transcript, encoding="utf-8")

    raw_plan = call_openai_for_montages(transcript, args.model, args.max_montages, args.min_seg_sec, args.max_seg_sec)
    (out_dir / "raw_montage_plan.json").write_text(json.dumps(raw_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    montages = normalize_montages(raw_plan["montages"], en_cues, duration, args.max_montages, args.min_seg_sec, args.max_seg_sec)
    final_plan = {"source_video": str(video), "layout": args.layout, "montages": montages}
    (out_dir / "final_montage_plan.json").write_text(json.dumps(final_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "final_montage_plan.csv", montages)

    for i, m in enumerate(montages, start=1):
        print(f"\n{i}. score={m['viral_score']} | {m['title_en']} | hook: {m['hook_hi']} | total: {m['total_duration_sec']}s")
        for j, s in enumerate(m["segments"], start=1):
            print(f"   {j}) {seconds_to_srt_time(s['start_sec'])} -> {seconds_to_srt_time(s['end_sec'])} | {s.get('segment_role', '')}")

    if args.dry_run:
        print("\nDry run enabled. Not rendering videos.")
        return

    rendered = []
    for i, montage in enumerate(montages, start=1):
        rendered.append(str(render_montage(video, montage, out_dir, i, args.layout, args.hook_seconds, args.font_name, args.font_size, args.crf, args.preset)))
    (out_dir / "rendered_montages.txt").write_text("\n".join(rendered), encoding="utf-8")
    print("\nDONE.")


if __name__ == "__main__":
    main()
