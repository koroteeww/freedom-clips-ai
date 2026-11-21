from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from freedom_clips.ffmpeg import probe_media, render_montage_clip
from freedom_clips.models import Unit
from freedom_clips.text_overlay import make_text_overlay_png


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_montage_speed_keeps_audio_and_video_duration_aligned(tmp_path: Path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    units = {
        0: Unit(0, 0.0, 2.0, 2.0, "a"),
        1: Unit(1, 2.0, 4.0, 2.0, "b"),
        2: Unit(2, 4.0, 6.0, 2.0, "c"),
    }
    beats = [
        {"unit_id": 0, "effect": "none", "speed": 1.10},
        {"unit_id": 1, "effect": "punch_in", "speed": 1.10},
        {"unit_id": 2, "effect": "none", "speed": 1.10},
    ]
    hook = make_text_overlay_png("test", tmp_path / "hook.png", y_ratio=0.08, font_size=50)
    output = tmp_path / "out.mp4"
    render_montage_clip(
        video=source,
        out_path=output,
        beats=beats,
        units_by_id=units,
        hook_png=hook,
        hook_seconds=1.0,
        overlay_windows=[],
        cta_png=None,
        cta_seconds=0,
        layout="crop",
        video_codec="libx264",
        audio_codec="aac",
        crf=28,
        preset="ultrafast",
    )
    expected = 6.0 / 1.10
    info = probe_media(output)
    actual = float(info["format"]["duration"])
    assert abs(actual - expected) < 0.5
    types = {stream.get("codec_type") for stream in info["streams"]}
    assert "video" in types
    assert "audio" in types
