from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def _font_candidates(bold: bool = True) -> list[str]:
    win = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\seguisb.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\NotoSans-Regular.ttf",
    ]
    linux = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    mac = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    return win + linux + mac


def load_font(size: int, font_path: str | None = None, bold: bool = True):
    if font_path:
        path = Path(font_path)
        if not path.exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        return ImageFont.truetype(str(path), size=size)
    for candidate in _font_candidates(bold=bold):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_text_to_width(text: str, font, max_width: int, max_lines: int = 4) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    words = text.split()
    image = Image.new("RGBA", (16, 16))
    draw = ImageDraw.Draw(image)
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(lines)) < len(text):
        lines[-1] = lines[-1].rstrip("…") + "…"
    return lines


def make_text_overlay_png(
    text: str,
    out_path: Path,
    *,
    y_ratio: float,
    font_size: int,
    font_path: str | None = None,
    width: int = 1080,
    height: int = 1920,
    max_text_width: int = 920,
    max_lines: int = 4,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = load_font(font_size, font_path=font_path, bold=True)
    lines = wrap_text_to_width(text, font, max_text_width, max_lines=max_lines)
    if not lines:
        image.save(out_path)
        return out_path

    line_gap = max(8, int(font_size * 0.18))
    measurements = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [max(1, box[3] - box[1]) for box in measurements]
    line_widths = [max(1, box[2] - box[0]) for box in measurements]
    text_height = sum(line_heights) + line_gap * (len(lines) - 1)
    box_width = min(width - 70, max(line_widths) + 80)
    box_height = text_height + 62
    x0 = (width - box_width) // 2
    y0 = int(height * y_ratio)
    y0 = max(28, min(height - box_height - 28, y0))

    try:
        draw.rounded_rectangle(
            [x0, y0, x0 + box_width, y0 + box_height],
            radius=24,
            fill=(0, 0, 0, 168),
            outline=(255, 255, 255, 70),
            width=2,
        )
    except AttributeError:
        draw.rectangle([x0, y0, x0 + box_width, y0 + box_height], fill=(0, 0, 0, 168))

    y = y0 + 31
    for line, line_height in zip(lines, line_heights):
        box = draw.textbbox((0, 0), line, font=font)
        text_width = box[2] - box[0]
        x = (width - text_width) // 2
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 190))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height + line_gap

    image.save(out_path)
    return out_path
