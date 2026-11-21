# Changelog

## Unreleased

- Added `dub_align_video.py` and `freedom_clips/dub_align.py`: the dub drives the video timeline. Pauses are used first, then the dub is sped up and/or the video slowed within limits. Each video slice is measured after rendering and the audio cut to match, so A/V stays sample-accurate. The background audio is time-warped too, and the subtitles are re-timed.
- Merged the 0.2.0 package refactor into the GitHub repository.
- Removed `noimg.py` and `ai_montage_clips.py`; `ai_viral_montage_clips.py` replaces both (original footage only).
- Kept the dubbing/transcription tools (`realtime_dub_mp4.py`, `transcribe_ru_diarized_srt_openai.py`, `tretyakov_translate_tts_kdenlive.py`, `build_exact_dub_mp3_from_manifest.py`) unchanged.
- `run_ai_opusclip_simple.py` now takes `--video` and `--language` instead of hardcoded paths, and accepts any Whisper language code.
- Added `.env.example`; tightened `.gitignore` (secrets, plan cache, archives, scratch SRTs).
- **Breaking:** `ai_opusclip.py --en/--hi` replaced by `--srt/--secondary-srt`; 0.1.x plan JSON files are not accepted.

## 0.2.0 — 2026-09-10

- Made continuous clip planning topic-agnostic and language-agnostic.
- Replaced language-specific JSON fields with `title`, `hook_text`, `caption`, and `why_selected`.
- Removed generated-art logic from the production montage pipeline.
- Hooks now appear over original footage while original audio starts immediately.
- Added synchronized video `setpts` + audio `atempo` speed handling.
- Fixed timestamp rounding overflow.
- Added natural cue-boundary fitting and explicit overlap control.
- Added strict JSON Schema planning via the OpenAI Responses API.
- Added plan caching, plan-only mode, and render-only mode.
- Added timestamp-overlap alignment for optional secondary SRT files.
- Centralized Pillow text overlays and ffmpeg rendering.
- Added post-render duration validation.
- Added automated tests and agent-oriented project documentation.

## 0.1.0

Initial MVP release:

- `ai_opusclip.py`: OpusClip-like continuous viral clips from SRT
- `ai_montage_clips.py`: montage clips from 5–10 second speech moments
- `realtime_dub_mp4.py`: OpenAI Realtime Translation dubbing for MP4
- Hindi hooks and Hindi-first workflow
- ffmpeg-only rendering pipeline
