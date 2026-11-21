# Freedom Clips AI — Open-Source OpusClip Alternative in Python

**Freedom Clips AI** turns long videos, podcasts, interviews, lectures, speeches, webinars, and conference recordings into vertical short-form clips using **Python, OpenAI, SRT subtitles, Pillow, and ffmpeg**.

It is a lightweight, inspectable alternative to closed video clipping SaaS products. The clip-selection logic is **topic-agnostic and language-agnostic by default**: the model infers the subject, language, audience, and editorial tone from the supplied transcript.

> Long video + SRT → AI clip plan → local ffmpeg rendering → Shorts / Reels / TikTok-ready MP4 files.

## TL;DR

- Open-source OpusClip-style workflow.
- Creates **continuous viral clips** or **multi-segment montage clips**.
- Works with English, Russian, Hindi, Arabic, Chinese, and other Unicode-language SRT files.
- Uses the OpenAI Responses API for clip planning.
- Uses **original footage only** for montage rendering.
- Does **not** call an image-generation API.
- Caches LLM plans so re-rendering does not spend API money again.
- Supports `--plan-only` and offline `--render-only --plan ...` workflows.
- Produces MP4, JSON metadata, CSV plans, and sidecar SRT subtitles.
- Runs on Windows, Linux, and macOS when Python and ffmpeg are available.

## What problem does this solve?

Creators often need to convert a 20–120 minute source video into multiple vertical clips. Commercial AI clipping tools can be convenient, but they are usually closed, difficult to automate, and priced per minute or per generated asset.

Freedom Clips AI keeps the expensive part small:

1. subtitles are parsed locally;
2. the LLM selects or arranges the strongest source moments;
3. the plan is cached;
4. all video rendering, cropping, pacing, overlays, concatenation, and encoding happen locally with ffmpeg.

The project is useful for:

- podcast to Shorts automation;
- interview clip generation;
- lecture and course highlights;
- conference speaker clips;
- founder and expert content;
- multilingual social video workflows;
- YouTube Shorts generation from SRT;
- TikTok and Instagram Reels generation;
- open-source AI video editing experiments;
- automated content pipelines and agentic media workflows.

## Two clip modes

### 1. Continuous clips — `ai_opusclip.py`

This mode behaves like a simple open-source OpusClip alternative.

The LLM selects contiguous timestamp ranges from the source transcript. The program then snaps candidate ranges to natural subtitle boundaries, rejects excessive overlap, adds a short on-screen hook, converts the video to 1080×1920, and renders each clip locally.

Typical use case:

```text
60-minute podcast
        ↓
5–12 strong continuous excerpts
        ↓
30–90 second vertical clips
```

### 2. Montage clips — `ai_viral_montage_clips.py`

This mode creates faster short-form edits from multiple semantic units of the same source video.

Typical use case:

```text
long interview
   ↓
3–6 selected source beats
   ↓
15–25 second montage
```

Montage mode can remain chronological or intentionally reorder source beats. Every output plan records original source unit IDs and an `integrity_risk` score.

**Important:** montage generation uses the original source video plus local text overlays. There is no generated-art pipeline and no image-generation API cost.

## Repository facts for humans and agents

| Question | Answer |
|---|---|
| Primary purpose | Convert long-form video + SRT into vertical short-form clips |
| Main language | Python 3.10+ |
| Video engine | ffmpeg / ffprobe |
| LLM API | OpenAI Responses API |
| Default planning model | `gpt-5.6-luna` |
| Image generation | None |
| Input transcription | Existing `.srt` file |
| Continuous clip script | `ai_opusclip.py` |
| Montage clip script | `ai_viral_montage_clips.py` |
| Shared package | `freedom_clips/` |
| Plan cache | `.cache/plans/` |
| Automated tests | `pytest` |
| Output aspect ratio | 1080×1920 |
| License | MIT |

For machine-oriented project context, see [`llms.txt`](llms.txt) and [`AGENTS.md`](AGENTS.md).

## Requirements

- Python **3.10 or newer**
- `ffmpeg` and `ffprobe` available in `PATH`
- an OpenAI API key for planning new clips
- an SRT transcript that corresponds to the source video

You **do not need an API key to render an existing plan**.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/freedom-clips-ai.git
cd freedom-clips-ai
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

Check ffmpeg:

```bash
ffmpeg -version
ffprobe -version
```

Set your API key when you want to create a new plan.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Linux/macOS:

```bash
export OPENAI_API_KEY="sk-..."
```

## Quick start: continuous clips

Input files:

```text
interview.mp4
interview.srt
```

Plan and render:

```bash
python ai_opusclip.py \
  --video interview.mp4 \
  --srt interview.srt \
  --out clips
```

Plan first without encoding video:

```bash
python ai_opusclip.py \
  --video interview.mp4 \
  --srt interview.srt \
  --out clips \
  --plan-only
```

Then render later without another LLM call:

```bash
python ai_opusclip.py \
  --video interview.mp4 \
  --srt interview.srt \
  --out clips_rendered \
  --plan clips/final_clip_plan.json \
  --render-only
```

## Quick start: montage clips

```bash
python ai_viral_montage_clips.py \
  --video interview.mp4 \
  --srt interview.srt \
  --out montage_clips \
  --n-clips 3
```

Default mode preserves source order:

```bash
--edit-mode chronological
```

Allow deliberate montage reordering:

```bash
python ai_viral_montage_clips.py \
  --video interview.mp4 \
  --srt interview.srt \
  --out montage_clips \
  --n-clips 3 \
  --edit-mode montage
```

## Multilingual workflow

The primary SRT can be in any language supported by the selected model and local font stack.

You can optionally provide a secondary SRT for additional multilingual context:

```bash
python ai_opusclip.py \
  --video interview.mp4 \
  --srt transcript_en.srt \
  --secondary-srt transcript_hi.srt \
  --hook-language hi \
  --out clips
```

Secondary subtitles are aligned by timestamp overlap rather than by assuming both files have identical cue indexes.

Hook defaults:

```text
--source-language auto
--hook-language same
--editorial-style auto
```

Examples of explicit overrides:

```bash
--source-language ru
--hook-language en
--editorial-style "professional, concise, B2B"
--content-context "Audience: software founders and engineering leaders"
```

## Continuous clip CLI options

```text
--video PATH                 Source MP4/video file
--srt PATH                   Primary SRT transcript
--secondary-srt PATH         Optional second SRT
--out PATH                   Output directory
--model MODEL                OpenAI model ID
--min-clips N                Minimum LLM candidates
--max-clips N                Maximum final clips
--min-sec N                  Minimum clip duration
--max-sec N                  Maximum clip duration
--max-overlap-sec N          Allowed overlap between selected clips; default 0
--source-language VALUE      auto or explicit language
--hook-language VALUE        same or explicit language
--editorial-style VALUE      auto or custom style
--content-context TEXT       Optional creator/audience context
--layout crop|blur           Vertical composition
--plan-only                  Plan and validate only
--render-only                Render only; requires --plan
--plan PATH                  Existing JSON plan
--refresh-plan               Ignore plan cache and call the LLM again
--cache-dir PATH             LLM plan cache directory
--font-path PATH             Explicit local font file
--crf N                      H.264 quality setting
--preset VALUE               ffmpeg encoder preset
--verbose-ffmpeg             Print ffmpeg commands and stderr
```

## Montage clip CLI options

```text
--video PATH                 Source video
--srt PATH                   SRT transcript
--out PATH                   Output directory
--n-clips N                  Number of montage plans
--model MODEL                OpenAI model ID
--min-unit-sec N             Minimum semantic subtitle unit size
--max-unit-sec N             Maximum semantic subtitle unit size
--min-clip-sec N             Minimum montage duration
--max-clip-sec N             Maximum montage duration
--edit-mode chronological    Preserve source order
--edit-mode montage          Permit source-beat reordering
--content-context TEXT       Optional context for the planner
--layout crop|blur           Vertical composition
--plan-only                  Create plan only
--render-only --plan PATH    Render an existing plan with no LLM call
--refresh-plan               Bypass cached plan
```

Backward-friendly aliases are accepted for the old montage input names:

```text
--input-mp4
--input_mp4
--outdir
--n_clips
```

## Output structure

Continuous mode:

```text
clips/
├── transcript_for_planning.txt
├── raw_plan.json
├── final_clip_plan.json
├── final_clip_plan.csv
├── rendered_files.txt
├── 01_093_example_title/
│   ├── 01_example_title.mp4
│   ├── clip.srt
│   ├── hook.png
│   └── metadata.json
└── ...
```

Montage mode:

```text
montage_clips/
├── merged_units.json
├── raw_plan.json
├── final_montage_plan.json
├── manifest.json
├── 01_094_example_title/
│   ├── 01_example_title.mp4
│   ├── clip.srt
│   ├── metadata.json
│   └── overlays/
│       ├── hook.png
│       ├── beat_01.png
│       └── cta.png
└── ...
```

## Cost controls

Freedom Clips AI is designed so repeated editing does not repeatedly call an LLM.

### Plan cache

A plan fingerprint includes:

- transcript or semantic units;
- prompt version;
- model;
- clip-count and duration settings;
- language/style/context settings.

Cached plans are stored under:

```text
.cache/plans/
```

Use:

```bash
--refresh-plan
```

only when you intentionally want a new model response.

### Offline rendering

Once a final plan exists, use:

```bash
--render-only --plan path/to/plan.json
```

No OpenAI API request is made in that workflow.

## Correct A/V speed handling

Montage beats may use a small speed value between `1.0` and `1.18`.

The renderer applies matching transformations to both streams:

```text
video → setpts=(PTS-STARTPTS)/speed
audio → atempo=speed
```

The generated sidecar SRT uses the same `unit.duration / speed` timing, and the final file duration is checked with `ffprobe` after rendering.

## Natural clip boundaries

Continuous-mode LLM timestamps are treated as editorial suggestions, not trusted frame cuts.

Before rendering, the validator:

1. searches nearby subtitle cue starts;
2. searches real subtitle cue ends;
3. finds a pair that satisfies the requested duration range;
4. rejects candidates that cannot be fitted naturally;
5. removes excessive overlap;
6. clamps editorial scores to valid ranges.

This avoids cutting a sentence merely to force an exact duration.

## Text overlays and fonts

Text is rendered to transparent PNG files with Pillow and composited by ffmpeg. This avoids depending on ffmpeg `drawtext` or ASS/libass for the core hook/CTA workflow and is more predictable across Windows installations.

The default font resolver tries common Arial, Segoe UI, DejaVu Sans, and Noto font paths. For Arabic, Chinese, Hindi, or other scripts, you may get better results by supplying a suitable local font:

```bash
--font-path /path/to/NotoSans.ttf
```

## Architecture

```text
freedom_clips/
├── models.py              # Cue and Unit dataclasses
├── srt.py                 # SRT parsing, alignment and safe time formatting
├── llm.py                 # OpenAI Responses API + strict JSON + retry/fallback
├── plan_cache.py          # deterministic plan cache
├── text_overlay.py        # Pillow text rendering
├── ffmpeg.py              # ffmpeg/ffprobe commands and rendering
├── validation.py          # boundary, overlap and montage validation
└── prompts/
    ├── opusclip.py        # generic continuous-clip prompt/schema
    └── montage.py         # original-footage montage prompt/schema
```

The two top-level scripts are intentionally thin workflow/CLI entry points.

## Run tests

```bash
pytest -q
```

Tests do not call paid APIs.

The suite covers:

- SRT parsing and Unicode;
- timestamp rounding around second/minute boundaries;
- generic prompt regression checks;
- no generated-art API regression checks;
- natural boundary fitting;
- overlap rules;
- montage plan validation;
- plan-cache determinism;
- an optional local ffmpeg A/V speed integration test.

## Privacy and data flow

Video rendering is local. The source video itself is not uploaded by these scripts.

When planning a new set of clips, the transcript text or merged transcript units are sent to the configured OpenAI model. If that is not acceptable for a particular source, create a plan yourself and use `--render-only --plan ...`.

## What this project is not

Freedom Clips AI is not:

- a web application;
- a hosted SaaS;
- a video transcription engine;
- a face-tracking editor;
- a voice-cloning system;
- a generative-art pipeline;
- a guarantee that a clip will become viral.

It is a small, hackable clip-planning and local-rendering toolkit.

## Suggested GitHub topics

For discoverability, consider these repository topics:

```text
opusclip-alternative
ai-video-editor
shorts-generator
podcast-to-shorts
ffmpeg
openai
python
instagram-reels
tiktok-video
youtube-shorts
srt
video-automation
content-automation
```

## SEO / discovery phrases

The repository directly addresses searches such as:

- open source OpusClip alternative;
- free OpusClip alternative Python;
- podcast to YouTube Shorts generator;
- AI video clip generator from SRT;
- ffmpeg Shorts generator;
- OpenAI video clipping Python;
- automatic Reels generator;
- TikTok clip generator from podcast;
- multilingual Shorts generator;
- AI podcast clipper open source;
- long video to short clips Python;
- agentic video editing workflow.

These phrases describe actual project capabilities rather than separate landing-page claims.

## Dubbing, translation and transcription tools

These standalone scripts predate the 0.2.0 refactor. They work, but they do not use the shared `freedom_clips/` package yet.

| Script | What it does |
|---|---|
| `realtime_dub_mp4.py` | Dubs an MP4 into other languages using OpenAI Realtime Translation. This is translation, not voice cloning. |
| `transcribe_ru_diarized_srt_openai.py` | Video/audio → diarized SRT using OpenAI transcription. |
| `tretyakov_translate_tts_kdenlive.py` | SRT → translated SRT → per-cue TTS → Kdenlive-ready dub assets. |
| `build_exact_dub_mp3_from_manifest.py` | Assembles one timeline-exact dub MP3 from a TTS manifest. |

Examples:

```bash
python realtime_dub_mp4.py input.mp4 --languages spanish
python realtime_dub_mp4.py input.mp4 --languages arabic hindi --sleep-factor 1.0
python transcribe_ru_diarized_srt_openai.py --input talk.mp4 --outdir talk_asr --language ru
```

Run each script with `--help` to see all of its options. `realtime_dub_mp4.py` needs `websocket-client`, which is included in `requirements.txt`.

### Generating an SRT first

If you have no transcript yet, `run_ai_opusclip_simple.py` transcribes the video with Whisper, reuses an existing usable SRT, runs `ai_opusclip.py`, and can burn subtitles into the rendered clips:

```bash
python run_ai_opusclip_simple.py --video talk.mp4 --language en
```

## Development notes

Read [`AGENTS.md`](AGENTS.md) before making automated refactors. The design priorities are:

1. keep paid API usage explicit and cacheable;
2. never introduce generated-art calls into montage rendering;
3. preserve A/V timing correctness;
4. keep prompts topic- and language-neutral by default;
5. prefer small modules over framework complexity;
6. keep render-only mode independent from the OpenAI API.

## Version

This package is the refactored **0.2.0** implementation derived from the project code-review specification in [`docs/CODE_REVIEW_SPEC.md`](docs/CODE_REVIEW_SPEC.md).

## About the author

**Ivan Koroteev** is a software engineer, AI creator and public speaker from Moscow. He holds a BMSTU engineering degree in Computer Systems and Software and works across backend engineering, generative AI, media production and education. He also publishes AI-assisted music and media under the pseudonym **Aivan Karade**.

## Migrating from 0.1.x

- `ai_opusclip.py --en X --hi Y` is now `--srt X --secondary-srt Y`. The secondary SRT is optional and can be in any language.
- `noimg.py` and `ai_montage_clips.py` were removed. Use `ai_viral_montage_clips.py`, which renders original footage only. The old scripts remain in git history (tag `v0.1.0`).
- Old JSON plans with language-specific fields (`title_en`, `hook_hi`, `caption_en`, `why_viral`) are not accepted by the 0.2.0 validators. Regenerate them, or run with `--refresh-plan`.

## License

MIT. See [`LICENSE`](LICENSE).
