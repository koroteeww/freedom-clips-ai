from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .models import Unit


class FFmpegError(RuntimeError):
    pass


def command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd: list[str], *, cwd: Path | None = None, verbose: bool = False) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("$", command_string(cmd), flush=True)
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-35:])
        raise FFmpegError(f"ffmpeg command failed ({proc.returncode})\n$ {command_string(cmd)}\n{tail}")
    if verbose and proc.stderr.strip():
        print(proc.stderr, flush=True)
    return proc


def require_ffmpeg(*, need_encoder: bool = True) -> None:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH")
    if need_encoder and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


def probe_media(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout)


def media_duration(path: Path) -> float:
    data = probe_media(path)
    return float(data.get("format", {}).get("duration") or 0.0)


def has_audio(path: Path) -> bool:
    data = probe_media(path)
    return any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))


def validate_render_duration(path: Path, expected: float, tolerance: float = 0.5) -> float:
    actual = media_duration(path)
    if abs(actual - expected) > tolerance:
        raise FFmpegError(
            f"Rendered duration drifted: expected {expected:.3f}s, got {actual:.3f}s "
            f"(tolerance {tolerance:.3f}s): {path}"
        )
    return actual


def _effect_params(effect: str) -> tuple[float, int, str | None]:
    zoom = 1.0
    x_shift = 0
    eq = None
    if effect == "punch_in":
        zoom = 1.08
    elif effect == "contrast_boost":
        eq = "eq=contrast=1.08:saturation=1.07:brightness=0.01"
    elif effect == "crop_left":
        x_shift = -80
    elif effect == "crop_right":
        x_shift = 80
    elif effect == "flash_cut":
        eq = "eq=brightness=0.035:contrast=1.06"
    return zoom, x_shift, eq


def vertical_filter(src: str, dst: str, *, layout: str, effect: str = "none") -> list[str]:
    """Build a normalized 1080x1920/30fps graph from a labeled input."""
    zoom, x_shift, eq = _effect_params(effect)
    filters: list[str] = []

    if layout == "crop":
        sw = max(1080, int(round(1080 * zoom)))
        sh = max(1920, int(round(1920 * zoom)))
        chain = (
            f"{src}scale={sw}:{sh}:force_original_aspect_ratio=increase,"
            f"crop=1080:1920:x='max(0,min(iw-1080,(iw-1080)/2+{x_shift}))':"
            "y='max(0,min(ih-1920,(ih-1920)/2))'"
        )
        if eq:
            chain += f",{eq}"
        chain += ",setsar=1,fps=30,format=yuv420p" + dst
        filters.append(chain)
        return filters

    if layout == "blur":
        base = dst.strip("[]")
        fg0 = f"[{base}_fg0]"
        bg0 = f"[{base}_bg0]"
        fg = f"[{base}_fg]"
        bg = f"[{base}_bg]"
        filters.append(f"{src}split=2{fg0}{bg0}")
        filters.append(
            f"{bg0}scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,gblur=sigma=28{bg}"
        )
        fg_width = max(1080, int(round(1080 * zoom)))
        fg_chain = f"{fg0}scale={fg_width}:-2:force_original_aspect_ratio=decrease"
        if eq:
            fg_chain += f",{eq}"
        fg_chain += fg
        filters.append(fg_chain)
        filters.append(
            f"{bg}{fg}overlay=x='(W-w)/2+{x_shift}':y='(H-h)/2',"
            f"setsar=1,fps=30,format=yuv420p{dst}"
        )
        return filters

    raise ValueError(f"Unknown layout: {layout}")


def render_contiguous_clip(
    *,
    video: Path,
    out_path: Path,
    start: float,
    end: float,
    hook_png: Path,
    hook_seconds: float,
    layout: str,
    video_codec: str,
    audio_codec: str,
    crf: int,
    preset: str,
    verbose: bool = False,
) -> Path:
    duration = end - start
    graph = vertical_filter("[0:v]", "[base]", layout=layout, effect="none")
    graph += [
        "[1:v]format=rgba[hook]",
        f"[base][hook]overlay=0:0:enable='between(t,0,{min(hook_seconds, duration):.3f})'[vout]",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video),
        "-loop", "1", "-framerate", "30", "-i", str(hook_png),
        "-filter_complex", ";".join(graph),
        "-map", "[vout]", "-map", "0:a?",
        "-t", f"{duration:.3f}",
        "-c:v", video_codec, "-preset", preset, "-crf", str(crf),
        "-c:a", audio_codec, "-b:a", "160k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest",
        str(out_path),
    ]
    run(cmd, verbose=verbose)
    validate_render_duration(out_path, duration)
    return out_path


def render_montage_clip(
    *,
    video: Path,
    out_path: Path,
    beats: list[dict[str, Any]],
    units_by_id: dict[int, Unit],
    hook_png: Path,
    hook_seconds: float,
    overlay_windows: list[tuple[Path, float, float]],
    cta_png: Path | None,
    cta_seconds: float,
    layout: str,
    video_codec: str,
    audio_codec: str,
    crf: int,
    preset: str,
    verbose: bool = False,
) -> Path:
    if not beats:
        raise ValueError("No beats to render")
    if not has_audio(video):
        raise FFmpegError("Montage rendering currently requires a source audio stream")

    n = len(beats)
    filters: list[str] = []
    vsrcs = "".join(f"[vsrc{i}]" for i in range(n))
    asrcs = "".join(f"[asrc{i}]" for i in range(n))
    filters.append(f"[0:v]split={n}{vsrcs}")
    filters.append(f"[0:a]asplit={n}{asrcs}")

    concat_inputs: list[str] = []
    cursor = 0.0
    expected_duration = 0.0
    for i, beat in enumerate(beats):
        unit = units_by_id[int(beat["unit_id"])]
        speed = float(beat.get("speed", 1.0))
        segment_duration = unit.duration / speed
        expected_duration += segment_duration

        trimmed = f"[vtrim{i}]"
        normalized = f"[v{i}]"
        filters.append(
            f"[vsrc{i}]trim=start={unit.start:.3f}:end={unit.end:.3f},"
            f"setpts=(PTS-STARTPTS)/{speed:.6f}{trimmed}"
        )
        filters.extend(
            vertical_filter(trimmed, normalized, layout=layout, effect=str(beat.get("effect", "none")))
        )
        filters.append(
            f"[asrc{i}]atrim=start={unit.start:.3f}:end={unit.end:.3f},"
            f"asetpts=PTS-STARTPTS,atempo={speed:.6f},"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
        cursor += segment_duration

    filters.append("".join(concat_inputs) + f"concat=n={n}:v=1:a=1[vcat][acat]")

    cmd = ["ffmpeg", "-y", "-i", str(video)]
    overlay_inputs: list[tuple[int, Path, float, float, str]] = []

    next_input = 1
    cmd += ["-loop", "1", "-framerate", "30", "-i", str(hook_png)]
    overlay_inputs.append((next_input, hook_png, 0.0, min(hook_seconds, expected_duration), "hook"))
    next_input += 1

    for overlay_path, start, end in overlay_windows:
        cmd += ["-loop", "1", "-framerate", "30", "-i", str(overlay_path)]
        overlay_inputs.append((next_input, overlay_path, start, end, f"beat{next_input}"))
        next_input += 1

    if cta_png is not None and cta_seconds > 0:
        cta_start = max(0.0, expected_duration - cta_seconds)
        cmd += ["-loop", "1", "-framerate", "30", "-i", str(cta_png)]
        overlay_inputs.append((next_input, cta_png, cta_start, expected_duration, "cta"))

    current = "[vcat]"
    for index, _path, start, end, label in overlay_inputs:
        png_label = f"[{label}_png]"
        out_label = f"[{label}_out]"
        filters.append(f"[{index}:v]format=rgba{png_label}")
        filters.append(
            f"{current}{png_label}overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'{out_label}"
        )
        current = out_label
    filters.append(f"{current}format=yuv420p[vout]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[acat]",
        "-t", f"{expected_duration:.3f}",
        "-c:v", video_codec, "-preset", preset, "-crf", str(crf),
        "-c:a", audio_codec, "-b:a", "160k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest",
        str(out_path),
    ]
    run(cmd, verbose=verbose)
    validate_render_duration(out_path, expected_duration)
    return out_path
