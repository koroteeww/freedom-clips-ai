#!/usr/bin/env python3
"""
ai_viral_montage_clips.py

Creates short viral-style montage clips from a long speech video + SRT.
Pipeline:
1) Parse and merge SRT into semantic units (~3-10s)
2) Ask GPT to score units and build viral clip scenarios (15-25s)
3) Ask GPT where to use AI-generated images / effects
4) Generate AI images with gpt-image-1
5) Use ffmpeg to render hook image intro + original/AI beats
6) Concat, burn subtitles, add end CTA question

Requirements:
  pip install openai python-dotenv
  ffmpeg and ffprobe in PATH

Example:
  python ai_viral_montage_clips.py \
    --input_mp4 speech.mp4 \
    --srt en.srt \
    --outdir out \
    --n_clips 3
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


# ----------------------------- data models -----------------------------

@dataclass
class SrtEntry:
    start: float
    end: float
    text: str


@dataclass
class Unit:
    unit_id: int
    start: float
    end: float
    duration: float
    text: str


# ----------------------------- utils -----------------------------

def log(msg: str):
    print(msg, flush=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sec_to_srt_time(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    s = total % 60
    total //= 60
    m = total % 60
    h = total // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def run(cmd: List[str], cwd: Optional[Path] = None):
    pretty = " ".join(shlex.quote(c) for c in cmd)
    log(f"[CMD] {pretty}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def sanitize_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-\. ]+", "", s, flags=re.UNICODE).strip()
    s = s.replace(" ", "_")
    return s[:max_len] or "clip"


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# ----------------------------- SRT parsing -----------------------------

TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)


def srt_time_to_sec(s: str) -> float:
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def parse_srt(path: Path) -> List[SrtEntry]:
    raw = load_text(path).replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())
    result: List[SrtEntry] = []
    for block in blocks:
        lines = [x.strip("\ufeff") for x in block.split("\n") if x.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 1 if lines[0].isdigit() else 0
        m = TIME_RE.match(lines[time_line_idx])
        if not m:
            continue
        start = srt_time_to_sec(m.group(1))
        end = srt_time_to_sec(m.group(2))
        text = " ".join(lines[time_line_idx + 1:]).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            result.append(SrtEntry(start=start, end=end, text=text))
    return result


def merge_entries(entries: List[SrtEntry], min_dur: float = 3.0, max_dur: float = 9.5) -> List[Unit]:
    units: List[Unit] = []
    if not entries:
        return units

    cur_texts: List[str] = []
    cur_start = entries[0].start
    cur_end = entries[0].end

    def flush():
        nonlocal cur_texts, cur_start, cur_end
        if not cur_texts:
            return
        text = " ".join(cur_texts).strip()
        unit = Unit(
            unit_id=len(units),
            start=cur_start,
            end=cur_end,
            duration=round(cur_end - cur_start, 3),
            text=text,
        )
        units.append(unit)
        cur_texts = []

    for e in entries:
        if not cur_texts:
            cur_texts = [e.text]
            cur_start = e.start
            cur_end = e.end
            continue

        proposed_end = e.end
        proposed_dur = proposed_end - cur_start

        strong_break = (
            cur_texts and (
                cur_texts[-1].endswith((".", "?", "!", "…")) or
                e.text[:1].isupper()
            )
        )

        if proposed_dur <= max_dur and not (proposed_dur >= min_dur and strong_break):
            cur_texts.append(e.text)
            cur_end = e.end
        else:
            flush()
            cur_texts = [e.text]
            cur_start = e.start
            cur_end = e.end

    flush()
    return units


# ----------------------------- OpenAI helpers -----------------------------

def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_2")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def call_gpt_json(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.choices[0].message.content
    return json.loads(text)


def generate_image(client: OpenAI, model: str, prompt: str, out_path: Path, size: str = "1024x1536"):
    log(f"[IMG] {prompt}")
    img = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
    )
    b64 = img.data[0].b64_json
    out_path.write_bytes(base64.b64decode(b64))


# ----------------------------- scenario prompting -----------------------------

def build_scenario_prompt(units: List[Unit], n_clips: int, max_ai_images_inside: int) -> Tuple[str, str]:
    system = (
        "You are a world-class short-form viral video strategist and editor. "
        "You design montage clips from a long speech. Output ONLY strict JSON. "
        "Optimize for: anti-skip hook in 0-3s, watchthrough, rewatch, shares, comments, follows. "
        "Duration target per clip: 15-25 seconds. "
        "Use the original speech audio. The speaker is on stage and visually boring, so you may replace some beats with AI-generated symbolic images. "
        "Use 1 AI hook image at the start of every clip, plus 0 to " + str(max_ai_images_inside) + " AI-image beats inside the clip only when it genuinely improves visual variety. "
        "You must choose from these effects only: none, punch_in, contrast_boost, crop_left, crop_right, slow_zoom, pan_left, pan_right, flash_cut. "
        "Hook image should be dramatic, symbolic, social-media-friendly, clean, not cluttered. "
        "No copyrighted characters, no logos, no watermarks. "
        "Every beat must reference a unit_id from the provided units. "
        "Prefer conflict frames, contradiction, fear+proof, debate, or warning. "
        "End with a short CTA question that can trigger comments. "
    )

    units_payload = [asdict(u) for u in units]

    user = (
        "Create clip scenarios as JSON with this exact top-level structure:\n"
        "{\n"
        "  \"global_theme\": string,\n"
        "  \"clips\": [\n"
        "    {\n"
        "      \"clip_id\": 1,\n"
        "      \"title\": string,\n"
        "      \"why_it_can_work\": string,\n"
        "      \"predicted_scores\": {\"watchthrough\":0-100,\"rewatch\":0-100,\"share\":0-100,\"comment\":0-100,\"follow\":0-100,\"early_drop_risk\":0-100,\"integrity_risk\":0-100},\n"
        "      \"hook\": {\"duration\": float 1.5-3.0, \"text\": string, \"image_prompt\": string},\n"
        "      \"beats\": [\n"
        "        {\n"
        "          \"unit_id\": int,\n"
        "          \"visual_source\": \"original\" | \"ai_image\",\n"
        "          \"image_prompt\": string or \"\",\n"
        "          \"effect\": one of allowed effects,\n"
        "          \"speed\": float 1.0-1.18,\n"
        "          \"overlay\": short overlay text or \"\",\n"
        "          \"reason\": short explanation\n"
        "        }\n"
        "      ],\n"
        "      \"cta_question\": string\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Create exactly {n_clips} clips.\n"
        "Rules:\n"
        "- total spoken duration per clip (sum of chosen units / speed) plus hook duration should be 15-25 seconds\n"
        "- 3 to 6 beats per clip\n"
        "- first 0-3 sec hook text must be punchy and scroll-stopping\n"
        "- preserve original voice by using the chosen units' audio\n"
        "- some clips may reorder units if it creates a stronger viral argument, but coherence matters\n"
        "- not every beat should use AI image; default to original unless AI image clearly improves the beat\n"
        "- overlays should be short, readable, semantically aligned with the spoken line\n"
        "- output only JSON\n\n"
        "Speech units:\n" + json_dumps(units_payload)
    )
    return system, user


# ----------------------------- ffmpeg / overlay helpers -----------------------------

# IMPORTANT:
# On Windows some ffmpeg builds crash on drawtext/subtitles because Fontconfig
# cannot load config. To avoid that, this version NEVER uses drawtext or
# subtitles filters. Text is rendered by Pillow into transparent PNG overlays,
# and ffmpeg only overlays PNGs. Much more stable on Windows.

def require_pillow():
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required for Windows-safe text overlays. Run: pip install pillow")


def build_atempo_chain(speed: float) -> str:
    speed = clamp(speed, 0.5, 2.0)
    return f"atempo={speed:.4f}"


def pick_font(size: int):
    require_pillow()
    candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\seguisb.ttf",
        # Linux common
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # macOS common
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def wrap_text_to_width(text: str, font, max_width: int) -> List[str]:
    require_pillow()
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    words = text.split(" ")
    lines: List[str] = []
    cur = ""
    dummy = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(dummy)
    for w in words:
        test = w if not cur else cur + " " + w
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_text_overlay_png(
    text: str,
    out_path: Path,
    y_ratio: float,
    font_size: int = 62,
    width: int = 1080,
    height: int = 1920,
    max_text_width: int = 940,
):
    """Create transparent 1080x1920 PNG with readable boxed text."""
    require_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = pick_font(font_size)
    lines = wrap_text_to_width(text, font, max_text_width)
    if not lines:
        img.save(out_path)
        return

    line_gap = int(font_size * 0.22)
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [b[3] - b[1] for b in bboxes]
    text_heights = sum(line_heights) + line_gap * (len(lines) - 1)
    text_widths = [b[2] - b[0] for b in bboxes]
    box_w = min(max(text_widths) + 72, width - 80)
    box_h = text_heights + 54
    box_x = (width - box_w) // 2
    box_y = int(height * y_ratio)
    box_y = max(30, min(height - box_h - 30, box_y))

    # rounded rectangle if available
    try:
        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            radius=22,
            fill=(0, 0, 0, 150),
            outline=(255, 255, 255, 90),
            width=2,
        )
    except Exception:
        draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(0, 0, 0, 150))

    y = box_y + 27
    for line, h in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (width - tw) // 2
        # tiny shadow
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 180))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += h + line_gap

    img.save(out_path)


def image_motion_filter(effect: str) -> str:
    if effect == "pan_left":
        motion = "zoompan=z='1.08':x='(iw-iw/zoom)*(1-on/90)':y='(ih-ih/zoom)/2':d=1:s=1080x1920:fps=30"
    elif effect == "pan_right":
        motion = "zoompan=z='1.08':x='(iw-iw/zoom)*(on/90)':y='(ih-ih/zoom)/2':d=1:s=1080x1920:fps=30"
    else:
        motion = "zoompan=z='min(1.18,zoom+0.0015)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=30"
    return ",".join([
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920",
        motion,
        "format=rgba",
    ])


def original_video_base_filter(effect: str) -> str:
    extra = []
    fg_scale = 1.0
    x_expr = "(W-w)/2"

    if effect == "punch_in":
        fg_scale = 1.08
    elif effect == "contrast_boost":
        extra.append("eq=contrast=1.07:saturation=1.08:brightness=0.01")
    elif effect == "crop_left":
        x_expr = "(W-w)/2-60"
    elif effect == "crop_right":
        x_expr = "(W-w)/2+60"
    elif effect == "slow_zoom":
        fg_scale = 1.12
    elif effect == "flash_cut":
        extra.append("eq=brightness=0.03")

    fg_w = int(1080 * fg_scale)
    last = f"[bg][fg]overlay={x_expr}:(H-h)/2,setsar=1"
    if extra:
        last += "," + ",".join(extra)
    last += ",format=rgba[base]"
    return ";".join([
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,boxblur=25:10,crop=1080:1920[bg]",
        f"[0:v]scale={fg_w}:-2:force_original_aspect_ratio=decrease[fg]",
        last,
    ])


def overlay_filter(base_filter: str) -> str:
    return base_filter + ";[1:v]format=rgba[ov];[base][ov]overlay=0:0,format=yuv420p[v]"


def render_hook_segment(image_path: Path, out_path: Path, duration: float, text: str):
    overlay_png = out_path.with_suffix(".overlay.png")
    make_text_overlay_png(text, overlay_png, y_ratio=0.08, font_size=70)
    fc = image_motion_filter("slow_zoom") + "[base];[1:v]format=rgba[ov];[base][ov]overlay=0:0,format=yuv420p[v]"
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{duration:.3f}", "-i", str(image_path),
        "-i", str(overlay_png),
        "-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex", fc,
        "-map", "[v]", "-map", "2:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-shortest",
        str(out_path),
    ]
    run(cmd)


def render_original_beat(input_mp4: Path, out_path: Path, start: float, duration: float, effect: str, speed: float, overlay_text: str):
    overlay_png = out_path.with_suffix(".overlay.png")
    make_text_overlay_png(overlay_text, overlay_png, y_ratio=0.78, font_size=54)
    fc = overlay_filter(original_video_base_filter(effect))
    atempo = build_atempo_chain(speed)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(input_mp4),
        "-i", str(overlay_png),
        "-filter_complex", fc,
        "-map", "[v]", "-map", "0:a?", "-af", atempo,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    run(cmd)


def render_ai_image_beat(input_mp4: Path, image_path: Path, out_path: Path, start: float, duration: float, effect: str, speed: float, overlay_text: str):
    overlay_png = out_path.with_suffix(".overlay.png")
    make_text_overlay_png(overlay_text, overlay_png, y_ratio=0.78, font_size=54)
    base = image_motion_filter(effect if effect in {"pan_left", "pan_right", "slow_zoom"} else "slow_zoom") + "[base]"
    fc = base + ";[2:v]format=rgba[ov];[base][ov]overlay=0:0,format=yuv420p[v]"
    atempo = build_atempo_chain(speed)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{duration:.3f}", "-i", str(image_path),
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(input_mp4),
        "-i", str(overlay_png),
        "-filter_complex", fc,
        "-map", "[v]", "-map", "1:a:0",
        "-af", atempo,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-shortest",
        str(out_path),
    ]
    run(cmd)


def concat_mp4s(parts: List[Path], out_path: Path):
    """
    Robust concat for Windows/MP4.

    The old version used the concat demuxer with -c copy. That is fast, but
    fragile for MP4 fragments made by different filter graphs: timestamps,
    time bases, and audio/video stream durations can be slightly different.
    Result: final file can report a much longer audio/video duration than the
    sum of visible parts.

    This version uses the concat FILTER and re-encodes once. It resets PTS for
    every part, forces 1080x1920 / 30fps / yuv420p, and normalizes audio to
    48k stereo AAC-compatible format.
    """
    if not parts:
        raise ValueError("concat_mp4s got no parts")

    cmd = ["ffmpeg", "-y"]
    for p in parts:
        cmd += ["-i", str(p)]

    n = len(parts)
    filters = []
    concat_inputs = []

    for i in range(n):
        filters.append(
            f"[{i}:v]setpts=PTS-STARTPTS,"
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,setsar=1,fps=30,format=yuv420p"
            f"[v{i}]"
        )
        filters.append(
            f"[{i}:a]asetpts=PTS-STARTPTS,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
            f"[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    filter_complex = ";".join(filters) + ";" + "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[v][a]"

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    run(cmd)


def burn_subtitles_and_cta(input_mp4: Path, input_srt: Path, out_path: Path, cta_question: str):
    """
    Windows-safe finalization.

    The old version used ffmpeg subtitles/drawtext filters, which can crash on
    Windows with Fontconfig errors. This version only overlays a CTA PNG for
    the last ~2.6 seconds. The detailed subtitles are still saved as final.srt,
    and short keyword overlays are already burned into each beat.
    """
    if not cta_question:
        run(["ffmpeg", "-y", "-i", str(input_mp4), "-c", "copy", str(out_path)])
        return

    dur = ffprobe_duration(input_mp4)
    enable_start = max(0.0, dur - 2.6)
    cta_png = out_path.with_suffix(".cta.png")
    make_text_overlay_png(cta_question, cta_png, y_ratio=0.10, font_size=54)
    fc = f"[0:v]format=rgba[base];[1:v]format=rgba[ov];[base][ov]overlay=0:0:enable='between(t,{enable_start:.2f},{dur:.2f})',format=yuv420p[v]"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_mp4), "-i", str(cta_png),
        "-filter_complex", fc,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    run(cmd)


# ----------------------------- subtitle assembly -----------------------------

def build_final_srt(clip: Dict[str, Any], units_by_id: Dict[int, Unit], out_path: Path):
    lines = []
    idx = 1
    cursor = float(clip["hook"]["duration"])

    for beat in clip["beats"]:
        u = units_by_id[int(beat["unit_id"])]
        speed = float(beat.get("speed", 1.0))
        beat_duration = (u.duration / speed)
        start = cursor
        end = cursor + beat_duration
        lines.append(str(idx))
        lines.append(f"{sec_to_srt_time(start)} --> {sec_to_srt_time(end)}")
        lines.append(u.text)
        lines.append("")
        idx += 1
        cursor = end

    save_text(out_path, "\n".join(lines).strip() + "\n")


# ----------------------------- main logic -----------------------------

def normalize_scenarios(raw: Dict[str, Any], units_by_id: Dict[int, Unit], max_ai_images_inside: int) -> Dict[str, Any]:
    clips = raw.get("clips", [])
    norm = {"global_theme": raw.get("global_theme", ""), "clips": []}
    allowed_effects = {"none", "punch_in", "contrast_boost", "crop_left", "crop_right", "slow_zoom", "pan_left", "pan_right", "flash_cut"}

    for i, clip in enumerate(clips, 1):
        hook = clip.get("hook", {})
        beats = clip.get("beats", [])
        ai_inside = 0
        clean_beats = []
        seen = set()
        for beat in beats:
            unit_id = int(beat["unit_id"])
            if unit_id not in units_by_id:
                continue
            effect = beat.get("effect", "none")
            if effect not in allowed_effects:
                effect = "none"
            visual_source = beat.get("visual_source", "original")
            if visual_source == "ai_image":
                ai_inside += 1
                if ai_inside > max_ai_images_inside:
                    visual_source = "original"
            clean_beats.append({
                "unit_id": unit_id,
                "visual_source": visual_source,
                "image_prompt": beat.get("image_prompt", "") or "",
                "effect": effect,
                "speed": clamp(float(beat.get("speed", 1.05)), 1.0, 1.18),
                "overlay": (beat.get("overlay", "") or "")[:60],
                "reason": beat.get("reason", ""),
            })
            seen.add(unit_id)

        if not clean_beats:
            continue

        norm["clips"].append({
            "clip_id": clip.get("clip_id", i),
            "title": clip.get("title", f"Clip {i}"),
            "why_it_can_work": clip.get("why_it_can_work", ""),
            "predicted_scores": clip.get("predicted_scores", {}),
            "hook": {
                "duration": clamp(float(hook.get("duration", 2.2)), 1.5, 3.0),
                "text": (hook.get("text", "") or "Stop scrolling")[:80],
                "image_prompt": hook.get("image_prompt", "dramatic symbolic image about digital freedom") or "dramatic symbolic image about digital freedom",
            },
            "beats": clean_beats,
            "cta_question": (clip.get("cta_question", "") or "What do you think?")[:120],
        })
    return norm


def create_clip_assets(
    client: OpenAI,
    image_model: str,
    input_mp4: Path,
    clip: Dict[str, Any],
    units_by_id: Dict[int, Unit],
    clip_dir: Path,
    image_cache: Dict[str, Path],
):
    ensure_dir(clip_dir)
    parts_dir = clip_dir / "parts"
    images_dir = clip_dir / "images"
    ensure_dir(parts_dir)
    ensure_dir(images_dir)

    # Hook image
    hook_hash = stable_hash(clip["hook"]["image_prompt"])
    hook_img = images_dir / f"hook_{hook_hash}.png"
    if not hook_img.exists():
        generate_image(client, image_model, clip["hook"]["image_prompt"], hook_img)

    hook_part = parts_dir / "part_00_hook.mp4"
    render_hook_segment(
        image_path=hook_img,
        out_path=hook_part,
        duration=float(clip["hook"]["duration"]),
        text=clip["hook"]["text"],
    )

    part_paths = [hook_part]

    for idx, beat in enumerate(clip["beats"], start=1):
        unit = units_by_id[int(beat["unit_id"])]
        out_part = parts_dir / f"part_{idx:02d}.mp4"
        visual_source = beat.get("visual_source", "original")
        effect = beat.get("effect", "none")
        speed = float(beat.get("speed", 1.0))
        overlay = beat.get("overlay", "")

        if visual_source == "ai_image" and beat.get("image_prompt"):
            prompt = beat["image_prompt"]
            h = stable_hash(prompt)
            if h in image_cache:
                img_path = image_cache[h]
            else:
                img_path = images_dir / f"beat_{idx:02d}_{h}.png"
                if not img_path.exists():
                    generate_image(client, image_model, prompt, img_path)
                image_cache[h] = img_path
            render_ai_image_beat(
                input_mp4=input_mp4,
                image_path=img_path,
                out_path=out_part,
                start=unit.start,
                duration=unit.duration,
                effect=effect if effect in {"pan_left", "pan_right", "slow_zoom"} else "slow_zoom",
                speed=speed,
                overlay_text=overlay,
            )
        else:
            render_original_beat(
                input_mp4=input_mp4,
                out_path=out_part,
                start=unit.start,
                duration=unit.duration,
                effect=effect,
                speed=speed,
                overlay_text=overlay,
            )
        part_paths.append(out_part)

    raw_concat = clip_dir / "raw_concat.mp4"
    concat_mp4s(part_paths, raw_concat)

    final_srt = clip_dir / "final.srt"
    build_final_srt(clip, units_by_id, final_srt)

    final_video = clip_dir / f"{sanitize_filename(clip['title'])}.mp4"
    burn_subtitles_and_cta(raw_concat, final_srt, final_video, clip.get("cta_question", ""))

    return final_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mp4", required=True, type=Path)
    parser.add_argument("--srt", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--n_clips", type=int, default=3)
    parser.add_argument("--gpt_model", default="gpt-5.4-mini")
    parser.add_argument("--image_model", default="gpt-image-1")
    parser.add_argument("--min_unit_dur", type=float, default=2.0)
    parser.add_argument("--max_unit_dur", type=float, default=12.5)
    parser.add_argument("--max_ai_images_inside", type=int, default=4)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    client = get_client()

    log("[1/6] Parsing SRT...")
    entries = parse_srt(args.srt)
    units = merge_entries(entries, min_dur=args.min_unit_dur, max_dur=args.max_unit_dur)
    if not units:
        raise RuntimeError("No units parsed from SRT")
    units_by_id = {u.unit_id: u for u in units}
    save_text(args.outdir / "merged_units.json", json_dumps([asdict(u) for u in units]))
    log(f"Parsed {len(entries)} subtitle lines -> {len(units)} merged units")

    log("[2/6] Asking GPT for viral scenarios...")
    system_prompt, user_prompt = build_scenario_prompt(
        units=units,
        n_clips=args.n_clips,
        max_ai_images_inside=args.max_ai_images_inside,
    )
    raw = call_gpt_json(client, args.gpt_model, system_prompt, user_prompt)
    save_text(args.outdir / "raw_scenarios.json", json_dumps(raw))
    scenarios = normalize_scenarios(raw, units_by_id, args.max_ai_images_inside)
    save_text(args.outdir / "normalized_scenarios.json", json_dumps(scenarios))

    clips = scenarios.get("clips", [])
    if not clips:
        raise RuntimeError("GPT returned no valid clips")

    log("[3/6] Rendering clips...")
    image_cache: Dict[str, Path] = {}
    outputs = []
    for clip in clips:
        clip_id = int(clip.get("clip_id", len(outputs) + 1))
        title = clip.get("title", f"Clip {clip_id}")
        clip_dir = args.outdir / f"clip_{clip_id:02d}_{sanitize_filename(title, 40)}"
        log(f"\n=== Rendering clip {clip_id}: {title} ===")
        save_text(clip_dir.with_suffix(".scenario.json"), json_dumps(clip))
        ensure_dir(clip_dir)
        final_path = create_clip_assets(
            client=client,
            image_model=args.image_model,
            input_mp4=args.input_mp4,
            clip=clip,
            units_by_id=units_by_id,
            clip_dir=clip_dir,
            image_cache=image_cache,
        )
        outputs.append(str(final_path.resolve()))
        log(f"[OK] {final_path}")

    log("\n[4/6] Done.")
    manifest = {
        "input_mp4": str(args.input_mp4.resolve()),
        "srt": str(args.srt.resolve()),
        "outputs": outputs,
        "theme": scenarios.get("global_theme", ""),
    }
    save_text(args.outdir / "manifest.json", json_dumps(manifest))

    log("Generated clips:")
    for p in outputs:
        log(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        raise
