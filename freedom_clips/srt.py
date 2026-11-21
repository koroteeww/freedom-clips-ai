from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Cue, Unit

_TAG_RE = re.compile(r"<[^>]+>")
_TS_RE = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2})(?:[,.](\d+))?$")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class SrtQualityReport:
    ok: bool
    reason: str
    cue_count: int
    word_count: int
    unique_word_count: int
    top_word: str | None = None
    top_word_ratio: float = 0.0


def parse_srt_timestamp(value: str) -> float:
    value = value.strip()
    match = _TS_RE.match(value)
    if not match:
        raise ValueError(f"Bad SRT timestamp: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    frac = float(f"0.{fraction}") if fraction else 0.0
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + frac


def seconds_to_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(float(seconds) * 100)))
    cs = total_cs % 100
    total_seconds = total_cs // 100
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean_srt_text(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip()) if raw.strip() else []
    cues: list[Cue] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if "-->" in lines[0]:
            index = len(cues) + 1
            time_line = lines[0]
            text_lines = lines[1:]
        elif len(lines) >= 2 and "-->" in lines[1]:
            try:
                index = int(lines[0])
            except ValueError:
                index = len(cues) + 1
            time_line = lines[1]
            text_lines = lines[2:]
        else:
            continue

        start_raw, end_raw = [part.strip() for part in time_line.split("-->", 1)]
        start = parse_srt_timestamp(start_raw.split()[0])
        end = parse_srt_timestamp(end_raw.split()[0])
        text = clean_srt_text(" ".join(text_lines))
        if text and end > start:
            cues.append(Cue(index=index, start=start, end=end, text=text))

    return cues


def assess_srt_quality(
    cues: list[Cue],
    *,
    min_cues: int = 3,
    min_span_sec: float = 10.0,
    max_top_word_ratio: float = 0.72,
    min_unique_words_for_long_transcript: int = 8,
    long_transcript_words: int = 40,
) -> SrtQualityReport:
    if not cues:
        return SrtQualityReport(False, "parsed 0 usable cues", 0, 0, 0)

    start = cues[0].start
    end = cues[-1].end
    span = end - start
    if len(cues) < min_cues:
        return SrtQualityReport(False, f"too few cues: {len(cues)} cues", len(cues), 0, 0)
    if span < min_span_sec:
        return SrtQualityReport(
            False,
            f"too short for minimum span {min_span_sec:.3f}s: {span:.3f}s",
            len(cues),
            0,
            0,
        )

    words = [word.lower().strip("'") for cue in cues for word in _WORD_RE.findall(cue.text)]
    words = [word for word in words if word]
    word_count = len(words)
    unique_words = set(words)
    unique_word_count = len(unique_words)
    if word_count == 0:
        return SrtQualityReport(False, "no words found in cues", len(cues), 0, 0)

    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    top_word, top_count = max(counts.items(), key=lambda item: item[1])
    top_word_ratio = top_count / word_count

    if word_count >= long_transcript_words and unique_word_count < min_unique_words_for_long_transcript:
        return SrtQualityReport(
            False,
            (
                "low-information repetitive transcript: "
                f"{unique_word_count} unique words across {word_count} words"
            ),
            len(cues),
            word_count,
            unique_word_count,
            top_word,
            top_word_ratio,
        )

    if word_count >= long_transcript_words and top_word_ratio >= max_top_word_ratio:
        return SrtQualityReport(
            False,
            (
                "repetitive transcript: "
                f"word {top_word!r} is {top_word_ratio:.1%} of {word_count} words"
            ),
            len(cues),
            word_count,
            unique_word_count,
            top_word,
            top_word_ratio,
        )

    return SrtQualityReport(
        True,
        (
            f"{len(cues)} cues, span {span:.3f}s ({start:.3f}->{end:.3f}), "
            f"{word_count} words, {unique_word_count} unique words"
        ),
        len(cues),
        word_count,
        unique_word_count,
        top_word,
        top_word_ratio,
    )


def merge_cues(cues: list[Cue], min_dur: float = 2.0, max_dur: float = 12.5) -> list[Unit]:
    if not cues:
        return []
    if min_dur <= 0 or max_dur <= 0 or min_dur > max_dur:
        raise ValueError("Require 0 < min_dur <= max_dur")

    units: list[Unit] = []
    current: list[Cue] = []

    def flush() -> None:
        if not current:
            return
        start = current[0].start
        end = current[-1].end
        text = " ".join(c.text for c in current).strip()
        units.append(
            Unit(
                unit_id=len(units),
                start=start,
                end=end,
                duration=round(end - start, 3),
                text=text,
            )
        )
        current.clear()

    for cue in cues:
        if not current:
            current.append(cue)
            continue

        proposed_duration = cue.end - current[0].start
        prior_text = current[-1].text.rstrip()
        strong_boundary = prior_text.endswith((".", "?", "!", "…", ":", ";"))

        if proposed_duration > max_dur or (
            strong_boundary and current[-1].end - current[0].start >= min_dur
        ):
            flush()
        current.append(cue)

    flush()
    return units


def _overlap_seconds(a: Cue, b: Cue) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def align_secondary_text(primary: list[Cue], secondary: list[Cue]) -> list[str | None]:
    """Align secondary cues to primary cues by maximum time overlap.

    If no cue overlaps, a near-midpoint cue may be used when its midpoint is within
    1.5 seconds of the primary midpoint. Otherwise the alignment is left empty.
    """
    if not secondary:
        return [None] * len(primary)

    result: list[str | None] = []
    cursor = 0
    for cue in primary:
        while cursor < len(secondary) and secondary[cursor].end < cue.start - 2.0:
            cursor += 1

        candidates: list[tuple[float, float, Cue]] = []
        j = cursor
        primary_mid = (cue.start + cue.end) / 2
        while j < len(secondary) and secondary[j].start <= cue.end + 2.0:
            other = secondary[j]
            overlap = _overlap_seconds(cue, other)
            mid_distance = abs(((other.start + other.end) / 2) - primary_mid)
            candidates.append((overlap, -mid_distance, other))
            j += 1

        if not candidates:
            result.append(None)
            continue

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        overlap, neg_mid_distance, best = candidates[0]
        if overlap > 0 or -neg_mid_distance <= 1.5:
            result.append(best.text)
        else:
            result.append(None)
    return result


def build_timecoded_transcript(
    primary: list[Cue],
    secondary: list[Cue] | None = None,
    include_secondary: bool = True,
) -> str:
    aligned = (
        align_secondary_text(primary, secondary or [])
        if include_secondary and secondary
        else [None] * len(primary)
    )
    lines: list[str] = []
    for cue, secondary_text in zip(primary, aligned):
        stamp = f"[{seconds_to_srt_time(cue.start)} --> {seconds_to_srt_time(cue.end)}]"
        lines.append(f"{stamp} PRIMARY: {cue.text}")
        if secondary_text:
            lines.append(f"SECONDARY: {secondary_text}")
    return "\n".join(lines)


def _split_text_for_subtitles(text: str, max_chars: int = 84) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                proposed = f"{current} {word}".strip()
                if current and len(proposed) > max_chars:
                    chunks.append(current)
                    current = word
                else:
                    current = proposed
            continue
        proposed = f"{current} {sentence}".strip()
        if current and len(proposed) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = proposed
    if current:
        chunks.append(current)
    return chunks or [text]


def _split_timed_text(
    text: str,
    start: float,
    end: float,
    *,
    max_chars: int = 84,
    min_piece_sec: float = 1.0,
) -> list[tuple[float, float, str]]:
    chunks = _split_text_for_subtitles(text, max_chars=max_chars)
    duration = end - start
    if len(chunks) <= 1 or duration <= min_piece_sec:
        return [(start, end, text)]

    min_total = min_piece_sec * len(chunks)
    if duration < min_total:
        return [(start, end, text)]

    weights = [max(1, len(chunk)) for chunk in chunks]
    total_weight = sum(weights)
    pieces: list[tuple[float, float, str]] = []
    cursor = start
    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        if index == len(chunks) - 1:
            piece_end = end
        else:
            raw_dur = duration * weight / total_weight
            remaining_pieces = len(chunks) - index - 1
            max_piece_end = end - remaining_pieces * min_piece_sec
            piece_end = min(max_piece_end, max(cursor + min_piece_sec, cursor + raw_dur))
        pieces.append((cursor, piece_end, chunk))
        cursor = piece_end
    return pieces


def write_clip_srt(cues: Iterable[Cue], start: float, end: float, out_path: Path) -> None:
    lines: list[str] = []
    index = 1
    for cue in cues:
        if cue.end <= start or cue.start >= end:
            continue
        local_start = max(cue.start, start) - start
        local_end = min(cue.end, end) - start
        if local_end <= local_start:
            continue
        for piece_start, piece_end, piece_text in _split_timed_text(cue.text, local_start, local_end):
            if piece_end <= piece_start:
                continue
            lines.extend(
                [
                    str(index),
                    f"{seconds_to_srt_time(piece_start)} --> {seconds_to_srt_time(piece_end)}",
                    piece_text,
                    "",
                ]
            )
            index += 1
    out_path.write_text("\n".join(lines).rstrip() + ("\n" if lines else ""), encoding="utf-8")


def write_montage_srt(
    beats: list[dict], units_by_id: dict[int, Unit], out_path: Path
) -> float:
    lines: list[str] = []
    cursor = 0.0
    for index, beat in enumerate(beats, start=1):
        unit = units_by_id[int(beat["unit_id"])]
        speed = float(beat.get("speed", 1.0))
        duration = unit.duration / speed
        lines.extend(
            [
                str(index),
                f"{seconds_to_srt_time(cursor)} --> {seconds_to_srt_time(cursor + duration)}",
                unit.text,
                "",
            ]
        )
        cursor += duration
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return cursor
