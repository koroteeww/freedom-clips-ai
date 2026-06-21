# Freedom Clips AI — Open Source OpusClip & ElevenLabs Alternative for Hindi, Shorts, Reels and TikTok

**Freedom Clips AI** is a lightweight Python MVP for creators who want an open, scriptable alternative to expensive AI video tools.

It helps you turn a long speech, podcast, interview, or lecture into:

- **viral vertical clips** like OpusClip
- **montage clips** made from 5–10 second moments across the whole video
- **AI dubbing / translation videos** using OpenAI Realtime Translation
- **Hindi-first short-form content** for YouTube Shorts, Instagram Reels and TikTok

This project was created in the spirit of **digital freedom**, inspired by Pavel Durov’s message about personal sovereignty, privacy and independence from centralized platforms.

The goal is simple:

> Give creators, educators and independent media people a cheap, hackable, open-source workflow instead of paying huge SaaS fees for every minute of video.

---

## Why this exists

Many AI creator tools are powerful, but they become expensive very quickly.

- OpusClip-style tools are often closed SaaS products.
- ElevenLabs-style dubbing can become expensive for long videos.
- Hindi and multilingual workflows are often weak or missing.
- Creators cannot easily inspect, modify or automate the pipeline.

**Freedom Clips AI** gives you a Python-first workflow using only:

- `ffmpeg`
- OpenAI API
- SRT subtitles
- simple local folders
- no video editor required

It is not a polished SaaS. It is an MVP you can fork, modify and build on.

---

## Main use cases

### 1. Open Source OpusClip Alternative

Use `ai_opusclip.py` to analyze a long video transcript and generate 10–20 short vertical clips.

Input:

```text
input.mp4
en.srt
hi.srt
```

Output:

```text
clips_durov/
  01_95_Digital_Freedom/
    01_95_Digital_Freedom.mp4
    metadata.json
  final_clip_plan.csv
  final_clip_plan.json
```

The script asks OpenAI to score possible clips for:

- hook strength
- emotional tension
- standalone clarity
- novelty
- quotability
- Shorts/Reels/TikTok potential

It then uses ffmpeg to create vertical clips.

---

### 2. AI Montage Clip Generator

Use `ai_montage_clips.py` to create viral montage clips from different parts of the speech.

Instead of one continuous 30–90 second clip, each final video is made from several short 5–10 second moments.

Example:

```text
montage_01.mp4
  segment 1: 00:02:14–00:02:22
  segment 2: 00:08:41–00:08:49
  segment 3: 00:17:10–00:17:18
  segment 4: 00:24:02–00:24:11
```

This is useful for:

- motivational edits
- digital freedom manifestos
- “best ideas from this speech”
- guru-style short clips
- idea compilations

---

### 3. OpenAI Realtime Dubbing Alternative to ElevenLabs

Use `realtime_dub_mp4.py` to translate and dub MP4 video into other languages using OpenAI Realtime Translation.

Supported target examples:

- Arabic
- Hindi
- Chinese
- French
- Spanish

The workflow:

```text
MP4 video
→ extract 24 kHz mono PCM16 audio with ffmpeg
→ stream audio to OpenAI Realtime Translation
→ receive translated speech audio
→ mux translated audio back into MP4
```

This is not voice cloning. It is a cheap translation/dubbing workflow.

For local voice cloning, you can experiment with pyVideoTrans, XTTS-v2, F5-TTS, CosyVoice or clone-voice separately.

---

## SEO / AEO keywords

This project is relevant for:

- OpusClip alternative
- open source OpusClip alternative
- free OpusClip alternative
- AI Shorts generator
- YouTube Shorts generator from SRT
- TikTok clip generator
- Instagram Reels generator
- Hindi Shorts generator
- Hindi subtitle video generator
- AI viral clip generator
- podcast to shorts Python
- ffmpeg shorts generator
- ElevenLabs alternative
- open source ElevenLabs alternative
- AI dubbing Python
- OpenAI Realtime dubbing
- OpenAI translation video
- Pavel Durov digital freedom speech clips

---

## Features

- Analyze long SRT transcripts with OpenAI
- Select viral 30–90 second clips
- Select montage clips from different timestamps
- Burn Hindi hooks into the top part of vertical videos
- Export metadata JSON and CSV
- Create `1080x1920` vertical videos
- Support center-crop or blurred-background layout
- Work with videos that already have hardcoded subtitles
- Dub videos into multiple languages through OpenAI Realtime Translation
- Simple scripts, no database, no web server

---

## Requirements

- Python 3.10+
- ffmpeg and ffprobe in PATH
- OpenAI API key
- Existing SRT files

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Check ffmpeg:

```bash
ffmpeg -version
ffprobe -version
```

Set your OpenAI key.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Windows CMD:

```bat
set OPENAI_API_KEY=sk-...
```

Linux/macOS:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Recommended input structure

```text
project/
  input.mp4
  en.srt
  hi.srt
  ai_opusclip.py
  ai_montage_clips.py
  realtime_dub_mp4.py
```

For the original use case:

- the speech was in Hindi
- `en.srt` was used for analysis
- `hi.srt` was used for Hindi hooks and context
- the original video already had hardcoded Hindi subtitles

---

## Usage: generate normal viral clips

Dry run first:

```bash
python ai_opusclip.py \
  --video input.mp4 \
  --en en.srt \
  --hi hi.srt \
  --out clips_durov \
  --min-clips 10 \
  --max-clips 20 \
  --dry-run
```

Render clips:

```bash
python ai_opusclip.py \
  --video input.mp4 \
  --en en.srt \
  --hi hi.srt \
  --out clips_durov \
  --min-clips 10 \
  --max-clips 20 \
  --layout crop
```

If your hardcoded subtitles are cut off by center crop, use blurred background layout:

```bash
python ai_opusclip.py \
  --video input.mp4 \
  --en en.srt \
  --hi hi.srt \
  --out clips_durov \
  --layout blur
```

---

## Usage: generate montage clips

Dry run:

```bash
python ai_montage_clips.py \
  --video input.mp4 \
  --en en.srt \
  --hi hi.srt \
  --out montage_durov \
  --max-montages 5 \
  --dry-run
```

Render:

```bash
python ai_montage_clips.py \
  --video input.mp4 \
  --en en.srt \
  --hi hi.srt \
  --out montage_durov \
  --max-montages 5 \
  --layout crop
```

Blur layout:

```bash
python ai_montage_clips.py \
  --video input.mp4 \
  --en en.srt \
  --hi hi.srt \
  --out montage_durov \
  --max-montages 5 \
  --layout blur
```

---

## Usage: dub MP4 into other languages

All languages:

```bash
python realtime_dub_mp4.py input.mp4
```

Spanish only:

```bash
python realtime_dub_mp4.py input.mp4 --languages spanish
```

Arabic and Hindi:

```bash
python realtime_dub_mp4.py input.mp4 --languages arabic hindi
```

Faster testing:

```bash
python realtime_dub_mp4.py input.mp4 --languages spanish --sleep-factor 0.25
```

Safe realtime mode:

```bash
python realtime_dub_mp4.py input.mp4 --languages spanish --sleep-factor 1.0
```

---

## Layout modes

### `crop`

Creates a true 9:16 vertical center crop:

```text
1080x1920
```

Best for centered speaker videos.

### `blur`

Creates a blurred background and keeps the whole original frame visible.

Use this when:

- the video has hardcoded subtitles
- the speaker is not centered
- center crop cuts important visual information

---

## Output files

Normal clips:

```text
clips_durov/
  final_clip_plan.csv
  final_clip_plan.json
  raw_openai_plan.json
  transcript_for_gpt.txt
  01_94_Title/
    01_94_Title.mp4
    hook.ass
    metadata.json
```

Montage clips:

```text
montage_durov/
  final_montage_plan.csv
  final_montage_plan.json
  raw_montage_plan.json
  transcript_for_gpt.txt
  montage_01_95_Title/
    seg_01.mp4
    seg_02.mp4
    seg_03.mp4
    montage_01_95_Title.mp4
    metadata.json
```

Dubbing:

```text
input_dub_arabic.mp4
input_dub_hindi.mp4
input_dub_chinese.mp4
input_dub_french.mp4
input_dub_spanish.mp4
```

---

## Notes on Hindi support

This MVP was built specifically because Hindi workflows are often weaker in short-form AI clipping tools.

Hindi is used in three ways:

1. the original speech may be Hindi
2. `hi.srt` can be included in the GPT prompt
3. generated hooks can be written in Hindi / Devanagari

The scripts assume you already have SRT files. You can create them with:

- pyVideoTrans
- Whisper
- OpenAI transcription
- faster-whisper
- any subtitle editor

---

## Cost philosophy

This project is designed to reduce cost by avoiding minute-based creator SaaS billing.

Instead of uploading everything to a closed SaaS platform, you run local ffmpeg and pay only for OpenAI API calls.

The actual cost depends on:

- model
- transcript length
- number of languages
- realtime audio duration
- your OpenAI pricing tier

Always test with a 10–60 second clip before processing a 30-minute video.

---

## Limitations

- No face tracking yet
- No automatic silence removal yet
- No browser UI yet
- No direct voice cloning in the bundled scripts
- Realtime dubbing is not voice cloning
- Montage clips may need human editorial review
- The OpenAI-selected moments depend heavily on SRT quality
- Hardcoded subtitles cannot be moved unless you regenerate the video from clean source

---

## Roadmap ideas

- Face tracking with OpenCV
- Whisper fallback when SRT is missing
- Hindi subtitle regeneration
- Auto B-roll prompts
- Auto thumbnail generation
- Telegram bot approval flow
- Web UI
- Batch folder processing
- SQLite project database
- XTTS-v2 voice clone pipeline
- pyVideoTrans integration
- Upload-ready captions and hashtags
- A/B testing of hooks

---

## Disclaimer

This project is independent and not affiliated with OpusClip, ElevenLabs, OpenAI, Telegram, or Pavel Durov.

Use only videos and subtitles you have the right to process. Respect copyright, privacy and platform rules.

---

## License

MIT License. See `LICENSE`.

---

## Author note

Created as a practical open-source experiment in digital freedom:

> A creator should be able to analyze, cut, translate and publish their own ideas without being trapped by closed platforms or insane per-minute pricing.

Fork it. Improve it. Make it useful.
