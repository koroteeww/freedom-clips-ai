#!/usr/bin/env python3
"""
tretyakov_translate_tts_kdenlive.py

Pipeline:
  Russian diarized SRT -> English translated SRT/JSON -> per-fragment OpenAI TTS MP3
  -> timed/tempo-adjusted MP3 clips -> simple .kdenlive / MLT project.

Install:
  pip install --upgrade openai
  ffmpeg + ffprobe must be in PATH

PowerShell:
  $env:OPENAI_API_KEY="sk-..."

Recommended first test:
  python tretyakov_translate_tts_kdenlive.py ^
    --srt "D:\GLOBAL_Rayban_Meta\full--transcript.srt" ^
    --video "D:\GLOBAL_Rayban_Meta\tretyakov.mp4" ^
    --outdir "D:\GLOBAL_Rayban_Meta\_tretyakov_en_dub" ^
    --limit 20

Full run:
  python tretyakov_translate_tts_kdenlive.py ^
    --srt "D:\GLOBAL_Rayban_Meta\full--transcript.srt" ^
    --video "D:\GLOBAL_Rayban_Meta\tretyakov.mp4" ^
    --outdir "D:\GLOBAL_Rayban_Meta\_tretyakov_en_dub"

Outputs:
  translated_en.json
  translated_en_with_speakers.srt
  translated_en_clean.srt
  audio_raw/0001_SPEAKER_FEMALE_1.mp3
  audio_timed/0001_SPEAKER_FEMALE_1.mp3
  tts_manifest.json
  tretyakov_dub.kdenlive
  tretyakov_dub.mlt

Important:
  OpenAI public TTS voices are built-in voices. True custom voice cloning is only available
  to eligible customers. You can still map each speaker to a different built-in voice here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ----------------------------------------------------------------------
# EDITABLE GLOSSARY
# ----------------------------------------------------------------------
# Keep this close to the top so you can quickly edit historical terms.
# The translator will be explicitly told to preserve these renderings.
GLOSSARY: Dict[str, str] = {
    "Павел Михайлович Третьяков": "Pavel Mikhailovich Tretyakov",
    "Павел Третьяков": "Pavel Tretyakov",
    "Третьяков": "Tretyakov",
    "Сергей Михайлович Третьяков": "Sergei Mikhailovich Tretyakov",
    "Третьяковская галерея": "the Tretyakov Gallery",
    "Государственная Третьяковская галерея": "the State Tretyakov Gallery",
    "Лаврушинский переулок": "Lavrushinsky Lane",
    "Замоскворечье": "Zamoskvorechye",
    "Москва-река": "the Moskva River",
    "меценат": "patron of the arts",
    "меценаты": "patrons of the arts",
    "благотворитель": "philanthropist",
    "благотворители": "philanthropists",
    "купец": "merchant",
    "купечество": "merchant class",
    "предприниматель": "entrepreneur",
    "коллекционер": "collector",
    "Передвижники": "the Peredvizhniki",
    "Товарищество передвижных художественных выставок": "the Society for Traveling Art Exhibitions",
    "передвижные художественные выставки": "traveling art exhibitions",
    "икона": "icon",
    "иконы": "icons",
    "русская живопись": "Russian painting",
    "русское искусство": "Russian art",
    "Московское купеческое общество": "the Moscow Merchant Society",
    "музей предпринимателей, меценатов и благотворителей": "the Museum of Entrepreneurs, Patrons and Philanthropists",
}

# ----------------------------------------------------------------------
# EDITABLE SPEAKER -> VOICE MAP
# ----------------------------------------------------------------------
# Built-in OpenAI voices include:
# alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse, marin, cedar
#
# If your OpenAI organization has custom voices enabled, you can try placing a custom voice id
# in "voice"; otherwise use built-in voices.
VOICE_MAP: Dict[str, Dict[str, str]] = {
    "SPEAKER_FEMALE_1": {
        "voice": "shimmer",
        "instructions": "Speak in clear documentary English, warm and intelligent, respectful, measured, museum expert tone.",
    },
    "SPEAKER_FEMALE_2": {
        "voice": "nova",
        "instructions": "Speak in clear documentary English, lively but dignified, as a knowledgeable museum guide.",
    },
    "SPEAKER_FEMALE_3": {
        "voice": "coral",
        "instructions": "Speak in clear documentary English, calm, thoughtful, respectful, with natural pacing.",
    },
    "SPEAKER_MALE_1": {
        "voice": "onyx",
        "instructions": "Speak in clear documentary English, mature male narrator, calm authority, respectful to Russian culture and history.",
    },
    "SPEAKER_FEMALE_DAUGHTER": {
        "voice": "sage",
        "instructions": "Speak in clear English with a younger female tone, natural and sincere, not cartoonish.",
    },
    "DEFAULT_FEMALE": {
        "voice": "nova",
        "instructions": "Speak in clear documentary English, warm and respectful.",
    },
    "DEFAULT_MALE": {
        "voice": "onyx",
        "instructions": "Speak in clear documentary English, mature and respectful.",
    },
    "DEFAULT": {
        "voice": "marin",
        "instructions": "Speak in clear documentary English, neutral and respectful.",
    },
}


@dataclass
class SrtItem:
    index: int
    start: float
    end: float
    speaker: str
    ru: str
    en: str = ""


@dataclass
class AudioItem:
    index: int
    speaker: str
    start: float
    end: float
    target_duration: float
    en: str
    voice: str
    raw_mp3: str
    timed_mp3: str
    raw_duration: float
    final_duration: float
    tempo: float
    adjusted: bool


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_tool(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} is not in PATH. Install it or add it to PATH.")


def run(cmd: List[str]) -> None:
    pretty = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    log(f"[CMD] {pretty}")
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def srt_time_to_sec(s: str) -> float:
    s = s.strip()
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def sec_to_srt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    if ms >= 1000:
        total += 1
        ms -= 1000
    s = total % 60
    total //= 60
    m = total % 60
    h = total // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def sec_to_mlt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    if ms >= 1000:
        total += 1
        ms -= 1000
    s = total % 60
    total //= 60
    m = total % 60
    h = total // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def safe_file_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s.strip("_") or "speaker"


def parse_srt(path: Path, limit: Optional[int] = None) -> List[SrtItem]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())
    items: List[SrtItem] = []
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

    for block in blocks:
        lines = [x.strip() for x in block.split("\n") if x.strip()]
        if len(lines) < 2:
            continue

        try:
            index = int(lines[0])
            time_line = lines[1]
            text_lines = lines[2:]
        except ValueError:
            index = len(items) + 1
            time_line = lines[0]
            text_lines = lines[1:]

        m = time_re.search(time_line)
        if not m:
            continue

        start = srt_time_to_sec(m.group(1))
        end = srt_time_to_sec(m.group(2))

        text = clean_text(" ".join(text_lines))
        speaker = "SPEAKER_UNKNOWN"
        sm = re.match(r"^\[([^\]]+)\]\s*(.*)$", text)
        if sm:
            speaker = sm.group(1).strip()
            text = clean_text(sm.group(2))

        if not text:
            continue

        items.append(SrtItem(index=index, start=start, end=end, speaker=speaker, ru=text))

        if limit and len(items) >= limit:
            break

    return items


def glossary_for_prompt() -> str:
    return "\n".join([f"- {ru} => {en}" for ru, en in GLOSSARY.items()])


def batch_items(items: List[SrtItem], batch_size: int, max_chars: int) -> List[List[SrtItem]]:
    batches: List[List[SrtItem]] = []
    cur: List[SrtItem] = []
    cur_chars = 0

    for item in items:
        item_chars = len(item.ru)
        if cur and (len(cur) >= batch_size or cur_chars + item_chars > max_chars):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(item)
        cur_chars += item_chars

    if cur:
        batches.append(cur)

    return batches


def load_existing_translations(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = {}
        for x in data.get("items", data if isinstance(data, list) else []):
            if "index" in x and x.get("en"):
                result[int(x["index"])] = str(x["en"])
        return result
    except Exception:
        return {}


def save_translation_json(items: List[SrtItem], path: Path) -> None:
    ensure_dir(path.parent)
    payload = {
        "glossary": GLOSSARY,
        "items": [asdict(x) for x in items],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_srt(items: List[SrtItem], path: Path, with_speaker: bool = True, lang_field: str = "en") -> None:
    lines: List[str] = []
    for n, item in enumerate(items, 1):
        text = item.en if lang_field == "en" else item.ru
        text = clean_text(text)
        if with_speaker:
            text = f"[{item.speaker}] {text}"

        lines.append(str(n))
        lines.append(f"{sec_to_srt_time(item.start)} --> {sec_to_srt_time(item.end)}")
        lines.append(text)
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def translate_batch(client: OpenAI, model: str, batch: List[SrtItem]) -> Dict[int, str]:
    system = (
        "You are a careful Russian-to-English documentary subtitle translator. "
        "Translate with respect for Pavel Tretyakov, Russian culture, art history, and the dignity of the film. "
        "Do not summarize. Do not add facts. Keep the subtitle meaning faithful. "
        "Make English natural for voiceover dubbing: clear, spoken, not academic. "
        "Preserve speaker intent and tone. Preserve names and historical terms using the glossary. "
        "Return ONLY valid JSON."
    )

    user = {
        "task": "Translate these Russian SRT fragments into natural English for documentary dubbing.",
        "glossary": GLOSSARY,
        "rules": [
            "Return JSON with key 'items'.",
            "Each item must contain index and en.",
            "Do not include speaker labels inside en.",
            "Keep the English concise enough for dubbing inside the original timing.",
            "If a Russian line is fragmented, translate it as a natural fragment, not as a full essay.",
        ],
        "items": [
            {
                "index": item.index,
                "speaker": item.speaker,
                "start": sec_to_srt_time(item.start),
                "end": sec_to_srt_time(item.end),
                "ru": item.ru,
            }
            for item in batch
        ],
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)

    translations: Dict[int, str] = {}
    for row in data.get("items", []):
        try:
            idx = int(row["index"])
            en = clean_text(str(row["en"]))
            if en:
                translations[idx] = en
        except Exception:
            continue

    return translations


def translate_all(
    client: OpenAI,
    items: List[SrtItem],
    outdir: Path,
    model: str,
    batch_size: int,
    max_chars: int,
    resume: bool = True,
) -> List[SrtItem]:
    translation_json = outdir / "translated_en.json"
    existing = load_existing_translations(translation_json) if resume else {}

    for item in items:
        if item.index in existing:
            item.en = existing[item.index]

    pending = [x for x in items if not x.en]
    log(f"[TRANSLATE] total={len(items)}, existing={len(items)-len(pending)}, pending={len(pending)}")

    batches = batch_items(pending, batch_size=batch_size, max_chars=max_chars)

    for bi, batch in enumerate(batches, 1):
        log(f"[TRANSLATE] batch {bi}/{len(batches)} items={len(batch)}")
        translations = translate_batch(client, model, batch)

        for item in batch:
            if item.index in translations:
                item.en = translations[item.index]
            else:
                log(f"[WARN] Missing translation for item {item.index}; using Russian as placeholder")
                item.en = item.ru

        # Critical checkpoint after every translation batch.
        save_translation_json(items, translation_json)
        write_srt(items, outdir / "translated_en_with_speakers.partial.srt", with_speaker=True)
        write_srt(items, outdir / "translated_en_clean.partial.srt", with_speaker=False)

    save_translation_json(items, translation_json)
    write_srt(items, outdir / "translated_en_with_speakers.srt", with_speaker=True)
    write_srt(items, outdir / "translated_en_clean.srt", with_speaker=False)
    return items


def choose_voice(speaker: str) -> Tuple[str, str]:
    if speaker in VOICE_MAP:
        v = VOICE_MAP[speaker]
    elif "FEMALE" in speaker.upper():
        v = VOICE_MAP["DEFAULT_FEMALE"]
    elif "MALE" in speaker.upper():
        v = VOICE_MAP["DEFAULT_MALE"]
    else:
        v = VOICE_MAP["DEFAULT"]

    return v["voice"], v.get("instructions", "")


def generate_tts_mp3(client: OpenAI, model: str, item: SrtItem, out_path: Path, voice: str, instructions: str) -> None:
    ensure_dir(out_path.parent)
    text = clean_text(item.en)
    if not text:
        text = "."

    log(f"[TTS] {item.index:04d} {item.speaker} voice={voice}")

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out_path)


def atempo_chain(tempo: float) -> str:
    """
    ffmpeg atempo traditionally works best from 0.5 to 2.0 per filter.
    Chain filters to support bigger changes.
    """
    tempo = max(0.05, float(tempo))
    factors: List[float] = []

    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0

    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5

    factors.append(tempo)
    return ",".join([f"atempo={x:.6f}" for x in factors])


def fit_audio_to_duration(
    input_mp3: Path,
    output_mp3: Path,
    target_duration: float,
    threshold: float,
) -> Tuple[float, float, bool]:
    ensure_dir(output_mp3.parent)
    raw_duration = ffprobe_duration(input_mp3)

    if target_duration <= 0.2:
        target_duration = raw_duration

    diff = raw_duration - target_duration

    if abs(diff) <= threshold:
        shutil.copy2(input_mp3, output_mp3)
        final_duration = ffprobe_duration(output_mp3)
        return raw_duration, final_duration, False

    # output duration ~= input duration / tempo, so tempo = raw / target.
    tempo = raw_duration / target_duration
    af = atempo_chain(tempo)

    # Make duration exact-ish for Kdenlive by padding/trimming after tempo.
    # apad protects against tiny undershoots; atrim cuts tiny overshoots.
    af = f"{af},apad,atrim=0:{target_duration:.3f},asetpts=PTS-STARTPTS"

    run([
        "ffmpeg", "-y",
        "-i", str(input_mp3),
        "-vn",
        "-filter:a", af,
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(output_mp3),
    ])

    final_duration = ffprobe_duration(output_mp3)
    return raw_duration, final_duration, True


def generate_all_tts(
    client: OpenAI,
    items: List[SrtItem],
    outdir: Path,
    tts_model: str,
    threshold: float,
    resume: bool = True,
) -> List[AudioItem]:
    raw_dir = outdir / "audio_raw"
    timed_dir = outdir / "audio_timed"
    ensure_dir(raw_dir)
    ensure_dir(timed_dir)

    manifest_path = outdir / "tts_manifest.json"
    audio_items: List[AudioItem] = []

    for n, item in enumerate(items, 1):
        if not item.en.strip():
            continue

        target = max(0.25, item.end - item.start)
        file_base = f"{n:04d}_{safe_file_part(item.speaker)}"
        raw_mp3 = raw_dir / f"{file_base}.mp3"
        timed_mp3 = timed_dir / f"{file_base}.mp3"

        voice, instructions = choose_voice(item.speaker)

        if not (resume and raw_mp3.exists()):
            generate_tts_mp3(client, tts_model, item, raw_mp3, voice, instructions)
        else:
            log(f"[TTS RESUME] raw exists: {raw_mp3.name}")

        if not (resume and timed_mp3.exists()):
            raw_dur, final_dur, adjusted = fit_audio_to_duration(raw_mp3, timed_mp3, target, threshold=threshold)
        else:
            raw_dur = ffprobe_duration(raw_mp3)
            final_dur = ffprobe_duration(timed_mp3)
            adjusted = abs(raw_dur - target) > threshold
            log(f"[AUDIO RESUME] timed exists: {timed_mp3.name}")

        tempo = raw_dur / target if target else 1.0

        audio_items.append(AudioItem(
            index=item.index,
            speaker=item.speaker,
            start=item.start,
            end=item.end,
            target_duration=target,
            en=item.en,
            voice=voice,
            raw_mp3=str(raw_mp3.resolve()),
            timed_mp3=str(timed_mp3.resolve()),
            raw_duration=raw_dur,
            final_duration=final_dur,
            tempo=tempo,
            adjusted=adjusted,
        ))

        # Critical checkpoint after every item.
        manifest_path.write_text(json.dumps([asdict(x) for x in audio_items], ensure_ascii=False, indent=2), encoding="utf-8")

    return audio_items


def add_prop(parent: ET.Element, name: str, value: Any) -> ET.Element:
    prop = ET.SubElement(parent, "property", {"name": name})
    prop.text = str(value)
    return prop


def make_producer(parent: ET.Element, producer_id: str, resource: Path, media_type: str, duration: Optional[float] = None) -> ET.Element:
    attrs = {"id": producer_id}
    if duration is not None:
        attrs["in"] = "00:00:00.000"
        attrs["out"] = sec_to_mlt_time(duration)

    prod = ET.SubElement(parent, "producer", attrs)
    add_prop(prod, "length", sec_to_mlt_time(duration or 0))
    add_prop(prod, "eof", "pause")
    add_prop(prod, "resource", str(resource.resolve()))
    add_prop(prod, "mlt_service", "avformat")
    add_prop(prod, "kdenlive:clipname", resource.name)
    add_prop(prod, "kdenlive:folderid", "")
    add_prop(prod, "kdenlive:id", producer_id.replace("producer", ""))
    add_prop(prod, "kdenlive:audio_track", "1" if media_type == "audio" else "0")
    add_prop(prod, "kdenlive:video_track", "1" if media_type == "video" else "0")
    return prod


def build_simple_kdenlive_project(
    video_path: Optional[Path],
    audio_items: List[AudioItem],
    out_path: Path,
    fps: int = 25,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """
    Creates a simple MLT XML project with:
      track 0: original video, audio hidden
      track 1: generated English MP3 fragments, video hidden

    Kdenlive .kdenlive files are MLT-based XML. This is intentionally simple and editable.
    Kdenlive may upgrade/enrich it when opened.
    """
    ensure_dir(out_path.parent)

    project_duration = 1.0
    if audio_items:
        project_duration = max(x.end for x in audio_items)
    if video_path and video_path.exists():
        try:
            project_duration = max(project_duration, ffprobe_duration(video_path))
        except Exception:
            pass

    root = ET.Element("mlt", {
        "LC_NUMERIC": "C",
        "version": "7.0.0",
        "producer": "main_bin",
        "root": str(out_path.parent.resolve()),
    })

    ET.SubElement(root, "profile", {
        "description": f"HD {width}x{height} {fps} fps",
        "width": str(width),
        "height": str(height),
        "progressive": "1",
        "sample_aspect_num": "1",
        "sample_aspect_den": "1",
        "display_aspect_num": "16",
        "display_aspect_den": "9",
        "frame_rate_num": str(fps),
        "frame_rate_den": "1",
        "colorspace": "709",
    })

    main_bin = ET.SubElement(root, "playlist", {"id": "main_bin"})
    add_prop(main_bin, "kdenlive:docproperties.version", "1.1")
    add_prop(main_bin, "kdenlive:docproperties.projectid", str(int(time.time())))
    add_prop(main_bin, "kdenlive:documentproperty.kdenliveversion", "generated-by-script")
    add_prop(main_bin, "kdenlive:documentproperty.locale", "en_US")

    producer_counter = 0

    video_producer_id = None
    if video_path:
        video_producer_id = f"producer{producer_counter}"
        producer_counter += 1
        make_producer(root, video_producer_id, video_path, "video", duration=project_duration)
        ET.SubElement(main_bin, "entry", {"producer": video_producer_id})

    audio_producer_ids: List[Tuple[str, AudioItem]] = []
    for ai in audio_items:
        pid = f"producer{producer_counter}"
        producer_counter += 1
        make_producer(root, pid, Path(ai.timed_mp3), "audio", duration=ai.final_duration)
        ET.SubElement(main_bin, "entry", {"producer": pid})
        audio_producer_ids.append((pid, ai))

    # Video track playlist
    video_playlist = ET.SubElement(root, "playlist", {"id": "playlist_video"})
    if video_producer_id:
        ET.SubElement(video_playlist, "entry", {
            "producer": video_producer_id,
            "in": "00:00:00.000",
            "out": sec_to_mlt_time(project_duration),
        })
    else:
        ET.SubElement(video_playlist, "blank", {"length": sec_to_mlt_time(project_duration)})

    # Audio track playlist with blanks to position fragments at SRT times
    audio_playlist = ET.SubElement(root, "playlist", {"id": "playlist_english_dub"})
    cursor = 0.0
    for pid, ai in sorted(audio_producer_ids, key=lambda x: x[1].start):
        if ai.start > cursor:
            ET.SubElement(audio_playlist, "blank", {"length": sec_to_mlt_time(ai.start - cursor)})

        ET.SubElement(audio_playlist, "entry", {
            "producer": pid,
            "in": "00:00:00.000",
            "out": sec_to_mlt_time(ai.target_duration),
        })
        cursor = max(cursor, ai.start + ai.target_duration)

    if cursor < project_duration:
        ET.SubElement(audio_playlist, "blank", {"length": sec_to_mlt_time(project_duration - cursor)})

    tractor = ET.SubElement(root, "tractor", {
        "id": "tractor0",
        "in": "00:00:00.000",
        "out": sec_to_mlt_time(project_duration),
    })
    add_prop(tractor, "kdenlive:projectTractor", "1")
    ET.SubElement(tractor, "track", {"producer": "playlist_video", "hide": "audio"})
    ET.SubElement(tractor, "track", {"producer": "playlist_english_dub", "hide": "video"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt", required=True, type=Path, help="Russian diarized SRT")
    parser.add_argument("--video", type=Path, default=None, help="Original video for Kdenlive timeline")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--translate_model", default="gpt-4.1-mini")
    parser.add_argument("--tts_model", default="gpt-4o-mini-tts")
    parser.add_argument("--batch_size", type=int, default=35)
    parser.add_argument("--max_batch_chars", type=int, default=7000)
    parser.add_argument("--duration_threshold", type=float, default=2.0, help="Adjust tempo when TTS differs from target by more than this many seconds")
    parser.add_argument("--limit", type=int, default=0, help="For testing: process only first N SRT items")
    parser.add_argument("--skip_translate", action="store_true")
    parser.add_argument("--skip_tts", action="store_true")
    parser.add_argument("--skip_kdenlive", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY") and not args.skip_translate and not args.skip_tts:
        raise RuntimeError("OPENAI_API_KEY is not set")

    require_tool("ffmpeg")
    require_tool("ffprobe")
    ensure_dir(args.outdir)

    items = parse_srt(args.srt, limit=args.limit or None)
    if not items:
        raise RuntimeError("No SRT items parsed")

    log(f"[SRT] Parsed {len(items)} items")
    speaker_counts: Dict[str, int] = {}
    for item in items:
        speaker_counts[item.speaker] = speaker_counts.get(item.speaker, 0) + 1
    log("[SPEAKERS] " + ", ".join(f"{k}={v}" for k, v in sorted(speaker_counts.items())))

    client = OpenAI()

    if args.skip_translate:
        data_path = args.outdir / "translated_en.json"
        if not data_path.exists():
            raise RuntimeError(f"--skip_translate but {data_path} does not exist")
        data = json.loads(data_path.read_text(encoding="utf-8"))
        by_index = {int(x["index"]): x for x in data.get("items", [])}
        for item in items:
            if item.index in by_index:
                item.en = by_index[item.index].get("en", "")
    else:
        items = translate_all(
            client=client,
            items=items,
            outdir=args.outdir,
            model=args.translate_model,
            batch_size=args.batch_size,
            max_chars=args.max_batch_chars,
            resume=not args.no_resume,
        )

    audio_items: List[AudioItem] = []
    if not args.skip_tts:
        audio_items = generate_all_tts(
            client=client,
            items=items,
            outdir=args.outdir,
            tts_model=args.tts_model,
            threshold=args.duration_threshold,
            resume=not args.no_resume,
        )
    else:
        manifest = args.outdir / "tts_manifest.json"
        if manifest.exists():
            audio_items = [AudioItem(**x) for x in json.loads(manifest.read_text(encoding="utf-8"))]

    if not args.skip_kdenlive and audio_items:
        kdenlive_path = args.outdir / "tretyakov_dub.kdenlive"
        mlt_path = args.outdir / "tretyakov_dub.mlt"
        build_simple_kdenlive_project(args.video, audio_items, kdenlive_path)
        build_simple_kdenlive_project(args.video, audio_items, mlt_path)
        log(f"[KDENLIVE] {kdenlive_path}")
        log(f"[MLT] {mlt_path}")

    log("[DONE]")


if __name__ == "__main__":
    main()
