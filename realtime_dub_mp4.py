import argparse
import base64
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import websocket


REALTIME_TRANSLATION_URL = (
    "wss://api.openai.com/v1/realtime/translations"
    "?model=gpt-realtime-translate"
)

SAMPLE_RATE = 24000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE * CHUNK_MS / 1000)

TARGET_LANGUAGES = {
    "arabic": "ar",
    "hindi": "hi",
    "chinese": "zh",
    "french": "fr",
    "spanish": "es",
}


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def extract_pcm24k(input_mp4: Path, output_pcm: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-i", str(input_mp4),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "s16le",
        str(output_pcm),
    ])


def pcm_to_wav(input_pcm: Path, output_wav: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-i", str(input_pcm),
        str(output_wav),
    ])


def mux_audio_to_video(input_mp4: Path, translated_wav: Path, output_mp4: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-i", str(input_mp4),
        "-i", str(translated_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_mp4),
    ])


def realtime_translate_pcm(
    source_pcm: Path,
    target_language_code: str,
    output_pcm: Path,
    source_transcript_txt: Path,
    target_transcript_txt: Path,
    realtime_sleep_factor: float = 1.0,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    headers = [
        f"Authorization: Bearer {api_key}",
        "OpenAI-Safety-Identifier: freedom-clips-ai-local-dubbing-script",
    ]

    ws = websocket.WebSocket()
    ws.connect(REALTIME_TRANSLATION_URL, header=headers)

    print(f"Connected. Target language: {target_language_code}")

    ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "audio": {
                "output": {
                    "language": target_language_code
                }
            }
        }
    }))

    closed = threading.Event()
    error_box: list[str] = []

    out_audio = open(output_pcm, "wb", buffering=0)
    src_txt = open(source_transcript_txt, "w", encoding="utf-8")
    tgt_txt = open(target_transcript_txt, "w", encoding="utf-8")

    def receive_loop():
        try:
            while not closed.is_set():
                raw = ws.recv()
                if not raw:
                    continue

                event = json.loads(raw)
                event_type = event.get("type")

                if event_type == "session.output_audio.delta":
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        out_audio.write(base64.b64decode(audio_b64))

                elif event_type == "session.output_transcript.delta":
                    delta = event.get("delta", "")
                    print(delta, end="", flush=True)
                    tgt_txt.write(delta)
                    tgt_txt.flush()

                elif event_type == "session.input_transcript.delta":
                    delta = event.get("delta", "")
                    src_txt.write(delta)
                    src_txt.flush()

                elif event_type == "error":
                    msg = json.dumps(event, ensure_ascii=False, indent=2)
                    error_box.append(msg)
                    print("\nRealtime error:", msg)
                    closed.set()

                elif event_type == "session.closed":
                    print("\nSession closed by server.")
                    closed.set()
                    break

        except Exception as e:
            error_box.append(str(e))
            closed.set()

    recv_thread = threading.Thread(target=receive_loop, daemon=True)
    recv_thread.start()

    chunk_seconds = CHUNK_MS / 1000.0

    try:
        with open(source_pcm, "rb") as f:
            sent_chunks = 0

            while not closed.is_set():
                chunk = f.read(CHUNK_BYTES)
                if not chunk:
                    break

                audio_b64 = base64.b64encode(chunk).decode("ascii")

                try:
                    ws.send(json.dumps({
                        "type": "session.input_audio_buffer.append",
                        "audio": audio_b64,
                    }))
                except Exception as e:
                    error_box.append(f"WebSocket send failed: {e}")
                    closed.set()
                    break

                sent_chunks += 1

                if sent_chunks % 100 == 0:
                    seconds_sent = sent_chunks * chunk_seconds
                    print(f"\nSent {seconds_sent:.1f}s audio...")

                if realtime_sleep_factor > 0:
                    time.sleep(chunk_seconds * realtime_sleep_factor)

        if closed.is_set():
            print("\nStopped sending because the session closed or failed.")
        else:
            print("\nSource audio finished. Sending session.close...")
            try:
                ws.send(json.dumps({"type": "session.close"}))
            except Exception as e:
                error_box.append(f"Could not send session.close: {e}")

            closed.wait(timeout=180)

    finally:
        closed.set()
        try:
            ws.close()
        except Exception:
            pass
        out_audio.close()
        src_txt.close()
        tgt_txt.close()

    if error_box:
        raise RuntimeError("Realtime translation failed:\n" + "\n".join(error_box[:3]))


def process_language(
    input_mp4: Path,
    source_pcm: Path,
    work_dir: Path,
    lang_name: str,
    lang_code: str,
    realtime_sleep_factor: float,
) -> None:
    print("\n" + "=" * 80)
    print(f"Processing {lang_name.upper()} / {lang_code}")
    print("=" * 80)

    translated_pcm = work_dir / f"translated_{lang_name}.pcm"
    translated_wav = work_dir / f"translated_{lang_name}.wav"
    src_txt = work_dir / f"source_transcript_for_{lang_name}.txt"
    tgt_txt = work_dir / f"target_transcript_{lang_name}.txt"
    output_mp4 = input_mp4.with_name(f"{input_mp4.stem}_dub_{lang_name}.mp4")

    realtime_translate_pcm(
        source_pcm=source_pcm,
        target_language_code=lang_code,
        output_pcm=translated_pcm,
        source_transcript_txt=src_txt,
        target_transcript_txt=tgt_txt,
        realtime_sleep_factor=realtime_sleep_factor,
    )

    pcm_to_wav(translated_pcm, translated_wav)
    mux_audio_to_video(input_mp4, translated_wav, output_mp4)

    print(f"\nDONE: {output_mp4}")
    print(f"Target transcript: {tgt_txt}")


def main():
    parser = argparse.ArgumentParser(description="Dub MP4 video into target languages using OpenAI Realtime Translation + ffmpeg.")
    parser.add_argument("input_mp4", help="Input MP4 video")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["arabic", "hindi", "chinese", "french", "spanish"],
        choices=list(TARGET_LANGUAGES.keys()),
        help="Target languages to generate",
    )
    parser.add_argument(
        "--sleep-factor",
        type=float,
        default=1.0,
        help="1.0 = realtime safest; 0.5 = 2x faster; 0.0 = fastest but may be unstable.",
    )
    args = parser.parse_args()

    input_mp4 = Path(args.input_mp4).resolve()
    if not input_mp4.exists():
        raise FileNotFoundError(input_mp4)

    work_dir = input_mp4.with_name(input_mp4.stem + "_realtime_work")
    work_dir.mkdir(exist_ok=True)

    source_pcm = work_dir / "source_24k_mono.pcm"

    print("Extracting source audio...")
    extract_pcm24k(input_mp4, source_pcm)

    for lang_name in args.languages:
        lang_code = TARGET_LANGUAGES[lang_name]
        process_language(input_mp4, source_pcm, work_dir, lang_name, lang_code, args.sleep_factor)


if __name__ == "__main__":
    main()
