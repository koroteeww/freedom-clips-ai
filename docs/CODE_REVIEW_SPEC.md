# Freedom Clips AI — Code Review & Improvement Specification

**Repository:** `koroteeww/freedom-clips-ai`  
**Review date:** 2026-09-10  
**Branch reviewed:** `main`

## Source snapshot

This review is based on these repository versions:

- `ai_opusclip.py` — blob `90c7e7fd82f08a3ae0a8bf9fe8b3654bff908fca`
- `ai_viral_montage_clips.py` — blob `a31ef19984f067de08907414094e6bbb84e14deb`
- `requirements.txt` — blob `da6d21a3df85a0ad1b84fe06877bae42c4726df8`
- `README.md` — blob `0968ceb0b7e9cd4fe34e775a9163941b5ac1d1fe`
- `noimg.py` — blob `1f8d2c653b959adf591c2b385aa7da68d1fee5b7`

---

# 1. Executive summary

The project is a useful, understandable MVP with a sensible core idea: use SRT as the semantic source, ask an LLM to choose clips, and let ffmpeg do the rendering.

The code is significantly better than a throwaway prototype because it already has:

- small helper functions rather than one giant `main`;
- dataclasses for subtitle data;
- JSON/CSV metadata outputs;
- validation/normalization after LLM output;
- CLI configuration;
- ffmpeg preflight checks in `ai_opusclip.py`;
- structured output in `ai_opusclip.py`;
- cross-platform awareness in `ai_viral_montage_clips.py`;
- reproducible output folders and manifests.

However, the current code is still tightly coupled to the original Hindi / digital-freedom use case, duplicates substantial functionality between scripts, has no automated tests, and has a few correctness and cost problems that should be fixed before adding more features.

## Quality scores

| Area | Score | Comment |
|---|---:|---|
| `ai_opusclip.py` | **6.5 / 10** | Good MVP structure and structured LLM output, but strongly hardcoded to Hindi/digital freedom and has brittle language/timestamp assumptions. |
| `ai_viral_montage_clips.py` | **5.5 / 10** | Readable and reasonably modular, but over-engineered around expensive generated images and contains an important A/V speed synchronization problem. |
| Repository overall | **6 / 10** | Useful open-source MVP, but not yet a clean reusable library/product architecture. |
| Target after this spec | **8–8.5 / 10** | Achievable without turning the project into a large framework. |

---

# 2. Mandatory change A — make `ai_opusclip.py` topic-agnostic

## Problem

`ai_opusclip.py` is presented as a general OpusClip-like generator, but its prompt and data model assume the original source video:

- “long Hindi speech about digital freedom”;
- motivational / visionary / guru-like editorial style;
- digital freedom, privacy, censorship, sovereignty, technology, courage;
- `hook_hi`;
- `title_en`;
- `caption_en`;
- required Hindi / Devanagari hook;
- CLI arguments `--en` and `--hi`;
- default font `Nirmala UI`;
- filename sanitization explicitly tailored to Latin, Cyrillic and Devanagari.

This means a business interview, scientific lecture, Russian podcast, Arabic speech, Chinese conference, comedy video, product tutorial, or political debate receives the wrong editorial framing.

## Required behavior

The default behavior must make **no assumptions about topic, ideology, language or editorial mood**.

The LLM should:

1. infer the main topic from the supplied transcript;
2. infer the source language;
3. infer a suitable short-form editorial style;
4. preserve the speaker’s meaning;
5. select clips that can stand on their own;
6. write hooks in the source language by default;
7. optionally use a user-selected hook language;
8. never introduce subject-specific instructions unless the user explicitly supplies them.

## New generic prompt

Recommended semantic instruction:

> You are an expert short-form video editor. Analyze the supplied timecoded transcript and select the strongest standalone clips for vertical short-form platforms. Infer the topic, audience, source language and appropriate editorial tone from the transcript itself. Do not assume any specific subject, ideology, country, industry or content style. Prefer clips with a strong opening, standalone clarity, emotional or intellectual tension, novelty, quotability, useful insight, surprise or a clear payoff. Preserve the speaker's meaning and do not invent facts.

The old digital-freedom-specific content should be deleted from the default prompt.

## Replace language-specific schema

Current concepts:

```text
title_en
hook_hi
caption_en
```

Target concepts:

```text
title
hook_text
caption
why_selected
```

Suggested clip schema:

```json
{
  "rank": 1,
  "start_sec": 123.4,
  "end_sec": 168.0,
  "viral_score": 87,
  "title": "Short descriptive title",
  "hook_text": "Short on-screen hook",
  "caption": "Optional social caption",
  "why_selected": "Reason this range works independently"
}
```

## Generic CLI

Recommended target:

```bash
python ai_opusclip.py \
  --video input.mp4 \
  --srt transcript.srt \
  --out clips \
  --min-clips 5 \
  --max-clips 12
```

Optional:

```text
--secondary-srt
--source-language auto
--hook-language same
--editorial-style auto
--content-context "optional free-form context"
```

For backward compatibility, `--en` and `--hi` can remain for one release as deprecated aliases, but all internal variables should become `primary_cues`, `secondary_cues`, etc.

## Generic Unicode filenames

Replace language-specific sanitization with Unicode-aware generic sanitization based on `\w`, or a small safe filename utility.

Do not explicitly whitelist only Devanagari/Cyrillic.

---

# 3. Mandatory change B — remove generated AI images from `ai_viral_montage_clips.py`

## Decision

Remove the image generation path **completely**, not merely disable it by default.

The project’s value proposition is cheap local video processing plus limited LLM planning. An image generation call for every hook and multiple beats works against that goal.

There is already a `noimg.py` experiment, but it still contains an optional `generate_image()` path. Maintaining multiple near-duplicate versions will increase bugs. The no-image architecture should become the main implementation instead.

## Remove

From `ai_viral_montage_clips.py`:

- `generate_image()`;
- calls to `client.images.generate(...)`;
- `gpt-image-1`;
- `--image_model`;
- `--max_ai_images_inside`;
- `visual_source`;
- `image_prompt`;
- image prompt instructions in the scenario prompt;
- hook-image generation;
- `render_ai_image_beat()`;
- generated image caches;
- image directories used only for generated art;
- `base64` import;
- `hashlib` / `stable_hash()` if no other cache uses them;
- `image_motion_filter()` if no local still-image feature remains.

Also remove documentation claiming that AI-generated images are part of the normal pipeline.

## Do NOT simply delete orchestration

`create_clip_assets()` currently does much more than create assets. It renders beats, concatenates them, builds the final SRT, and finalizes the video.

Therefore replace it with a clearer function such as:

```python
def render_clip(...):
    ...
```

or:

```python
def render_montage_clip(...):
    ...
```

Do not delete the orchestration responsibility without replacing it.

## Replace the AI hook image with an original-video hook

Best low-cost behavior:

- start immediately with the first selected original-video beat;
- show the LLM-generated hook text over the first 1.5–3 seconds;
- keep original audio playing immediately;
- optionally apply `punch_in`, crop, contrast or slow zoom to the first beat.

This is better than the current silent generated-image intro because it removes image cost and avoids spending the most valuable first seconds without the speaker’s voice.

### New hook model

Instead of:

```json
"hook": {
  "duration": 2.2,
  "text": "...",
  "image_prompt": "..."
}
```

use:

```json
"hook": {
  "display_seconds": 2.2,
  "text": "..."
}
```

The hook duration is now an **overlay duration**, not extra clip duration.

### New beat model

```json
{
  "unit_id": 42,
  "effect": "punch_in",
  "speed": 1.05,
  "overlay": "short keyword phrase",
  "reason": "..."
}
```

No `visual_source`. No `image_prompt`.

---

# 4. P0 correctness fixes

These should be implemented before architecture cleanup.

## P0.1 Fix video/audio speed synchronization in `ai_viral_montage_clips.py`

### Current problem

The original-video beat path applies:

```text
atempo=<speed>
```

to audio, while the video path does not equivalently change video PTS.

At the same time, `build_final_srt()` calculates:

```text
beat_duration = unit.duration / speed
```

So the subtitle timeline assumes both video and audio were accelerated.

This can produce:

- audio ending before video;
- silence at the end of beats;
- subtitle drift;
- concatenation duration inconsistencies;
- incorrect final CTA timing.

### Required fix

If speed is supported, apply the same effective speed to both streams.

Conceptually:

```text
video: setpts=PTS/<speed>
audio: atempo=<speed>
```

Then validate rendered beat duration against:

```text
unit.duration / speed
```

within a small tolerance.

Alternative simpler solution: remove per-beat speed entirely for v1.1 and keep `speed=1.0`.

I recommend fixing it properly rather than removing it.

## P0.2 Fix timestamp rounding overflow

Both scripts calculate fractional milliseconds/centiseconds independently of total rounded time.

Values near the next second can produce invalid timestamps such as:

```text
00:00:01,1000
```

or ASS centiseconds of `100`.

Convert the complete time to integer milliseconds/centiseconds first, then derive hours/minutes/seconds/remainder.

Add tests around:

```text
0
1.9994
1.9996
59.9996
3599.9996
```

## P0.3 Rename or fix `burn_subtitles_and_cta()`

The function name says that it burns subtitles, but the implementation only overlays the final CTA. `input_srt` is effectively not used for rendering subtitles.

Choose one:

### Preferred cheap/simple option

Rename:

```text
burn_subtitles_and_cta
```

to:

```text
finalize_with_cta
```

Remove the unused `input_srt` parameter and clearly document that `final.srt` is a sidecar subtitle file.

### Alternative

Actually burn subtitles.

If cross-platform ffmpeg/libass remains unreliable on Windows, reuse the Pillow-overlay strategy instead of claiming subtitles are burned when they are not.

## P0.4 Validate actual clip duration after rendering

After each final clip:

- call `ffprobe`;
- compare actual duration to planned duration;
- fail or warn if drift exceeds a tolerance, e.g. 0.25–0.5 seconds.

This would catch the current speed bug automatically.

---

# 5. `ai_opusclip.py` additional improvements

## P1.1 Improve clip boundary fitting

The current validation snaps start/end to nearby cues, but if duration is too short it can extend the end to an arbitrary `start + min_sec`. If too long it can cut to `start + max_sec`.

That can defeat the earlier rule to use natural thought boundaries.

Instead:

1. snap candidate start to cue start;
2. choose an end from real cue ends;
3. search nearby cue boundaries for the closest duration inside `[min_sec, max_sec]`;
4. reject the candidate if no acceptable natural boundary exists.

## P1.2 Enforce overlap policy explicitly

The prompt says clips should not overlap, while the implementation tolerates a few seconds.

Add:

```text
--max-overlap-sec 0
```

Default to zero.

If overlap is intentionally permitted, make it a documented setting.

## P1.3 Clamp and validate LLM scores

Normalize `viral_score` to `0..100`.

Validate:

- `start < end`;
- finite numeric timestamps;
- score range;
- non-empty title;
- non-empty hook;
- timestamps within source duration.

## P1.4 Better secondary subtitle alignment

Current secondary subtitles are associated mainly by list position plus a timestamp tolerance.

For generic multilingual inputs, align by time:

- find the secondary cue with maximum overlap;
- or nearest midpoint;
- skip secondary text when confidence is poor.

Do not assume both SRT files have identical cue counts/order.

## P1.5 Cross-platform hook rendering

`ai_opusclip.py` uses ffmpeg `subtitles=hook.ass`, while the montage script already contains a Pillow overlay approach specifically intended to avoid Windows Fontconfig/libass problems.

Unify on one text overlay implementation.

Recommended:

```text
text_overlay.py
```

with:

- font selection;
- Unicode-safe wrapping;
- hook overlay generation;
- CTA overlay generation.

This also removes the Hindi-specific default dependency on `Nirmala UI`.

## P1.6 Retry/fallback policy

The current structured-output call catches any exception and immediately makes a second LLM request using plain JSON.

That can double cost on rate limits, timeouts or networking failures.

Separate:

- retryable transport/rate-limit failures → bounded exponential retry;
- structured-output incompatibility / parsing failure → plain JSON fallback;
- invalid model/config → fail immediately.

---

# 6. `ai_viral_montage_clips.py` additional improvements

## P1.7 Strict structured output

`ai_opusclip.py` already has a JSON schema; `ai_viral_montage_clips.py` should use the same approach.

Define a strict clip scenario schema and validate before rendering.

The normalizer should be a safety layer, not the primary parser for arbitrary LLM JSON.

## P1.8 Add deterministic duration validation to plans

Before ffmpeg rendering:

```text
sum(unit.duration / speed)
```

must fall inside the requested clip range.

After the hook-image removal, hook display time does **not** add duration.

Reject or repair invalid plans before spending ffmpeg time.

## P1.9 Make reordering explicit

The current montage prompt allows units to be reordered to strengthen an argument.

That is useful for montage, but it can also change rhetorical context.

Add:

```text
--allow-reorder
```

Default recommendation:

```text
False
```

or expose modes:

```text
--edit-mode chronological
--edit-mode montage
```

For `chronological`, unit IDs must be increasing.

For `montage`, reordering is allowed but `integrity_risk` should remain low and metadata should record original timestamps.

## P1.10 Remove dead and misleading state

Examples to clean up after the no-image refactor:

- unused `seen`;
- unused imports;
- obsolete image cache;
- obsolete AI-image limits;
- progress labels `[1/6] ... [4/6]` that do not match actual stages.

---

# 7. Cost-control improvements

The repo exists partly to avoid expensive creator SaaS, so API cost should be a first-class design concern.

## P1.11 Cache LLM plans

Create a fingerprint from:

```text
transcript content
prompt version
model
clip duration settings
language/style options
```

Store:

```text
.cache/plans/<fingerprint>.json
```

Default behavior:

- reuse cache if present;
- `--refresh-plan` forces a new LLM request.

This makes repeated rendering/editing almost free.

## P1.12 Render from an existing plan

Add:

```text
--plan final_clip_plan.json
```

or:

```text
--reuse-plan
```

This allows changing crop, font, CRF, layout, effects, codec etc. without another LLM request.

## P1.13 Explicit planning mode

Use clearer modes:

```text
--plan-only
--render-only --plan ...
```

The current `--dry-run` still performs LLM planning, which is useful but not obvious from the name.

## P1.14 Control prompt size for long videos

A long full transcript sent in one request can become the main cost.

For long videos:

1. chunk transcript into windows;
2. ask for candidate ranges per chunk;
3. perform one compact global rerank over candidate summaries;
4. render only winners.

Do this only above a configurable token/character threshold so short videos retain the simple one-call path.

## P1.15 Lower unnecessary output budget

A clip plan rarely needs 12,000 output tokens.

Make output budget proportional to requested clip count and schema size.

---

# 8. Rendering performance and quality

## P2.1 Reduce repeated H.264 re-encoding

The montage pipeline currently renders individual beats, re-encodes during concatenation, and re-encodes again for CTA overlay.

That costs time and introduces generational quality loss.

Target architecture:

```text
source MP4
 -> one ffmpeg filter_complex per final clip
 -> trim selected ranges
 -> transform/crop/effects
 -> concat filters
 -> overlays
 -> CTA
 -> one final H.264 encode
```

This is more complex, so it is a P2 optimization, not the first refactor.

A safer intermediate step is to combine concat + CTA into the same final encoding pass.

## P2.2 Optional encoder configuration

Expose:

```text
--video-codec libx264
--audio-codec aac
--preset veryfast
--crf 20
```

Later add optional hardware encoders only if detected.

Do not make GPU encoders required.

## P2.3 Centralize ffmpeg invocation

Create one helper that:

- logs shell-safe command representation;
- captures stderr;
- raises a useful error containing the failed command and ffmpeg tail;
- optionally supports verbose mode.

---

# 9. Repository architecture

Do not create a huge framework. Extract only duplicated, stable concepts.

Recommended structure:

```text
freedom_clips/
  __init__.py
  models.py
  srt.py
  llm.py
  text_overlay.py
  ffmpeg.py
  validation.py
  prompts/
    opusclip.py
    montage.py

ai_opusclip.py
ai_viral_montage_clips.py
```

Responsibility:

### `models.py`

- subtitle cue/unit dataclasses;
- clip plan dataclasses / typed models.

### `srt.py`

- SRT parsing;
- timestamp formatting;
- subtitle alignment.

### `llm.py`

- OpenAI client creation;
- structured JSON calls;
- retry policy;
- plan cache.

### `text_overlay.py`

- font resolution;
- Unicode-safe wrapping;
- Pillow hook/CTA overlays.

### `ffmpeg.py`

- ffprobe;
- command runner;
- shared crop/blur filters;
- render validation.

### `validation.py`

- plan schema validation;
- duration checks;
- overlap checks;
- bounds checks.

The two top-level scripts should mainly contain CLI definitions and workflow orchestration.

---

# 10. Dependencies

Current `requirements.txt` does not contain Pillow, although the viral montage script requires Pillow for overlay PNGs.

At minimum:

```text
openai>=1.0.0
Pillow>=10
websocket-client>=1.7.0
```

Better: pin/test compatible upper/lower ranges through a release process rather than leaving dependencies indefinitely open-ended.

If `python-dotenv` is not actually imported, do not mention it as a requirement in file documentation.

---

# 11. Testing specification

Add `pytest` and create fast unit tests plus one tiny ffmpeg integration test.

## Unit tests

### SRT

- parses numbered SRT;
- parses SRT without numeric index;
- strips tags/BOM;
- handles Unicode;
- timestamp conversion around second/minute/hour boundaries.

### Clip validation

- rejects negative start;
- rejects end before start;
- clamps score;
- enforces max duration;
- finds natural cue boundaries;
- zero-overlap mode works.

### Generic prompt

Assert default prompt does **not** contain:

```text
digital freedom
privacy
censorship
guru-like
Hindi speech
hook_hi
Devanagari
```

### No-image regression

Assert production montage code does **not** contain or invoke:

```text
client.images.generate
gpt-image-1
image_prompt
visual_source == "ai_image"
```

### A/V speed

Given a synthetic 10-second clip rendered at `speed=1.10`:

- expected duration ≈ `9.09s`;
- audio/video durations should match within tolerance.

## Integration test

Generate a synthetic short video with ffmpeg (`testsrc` + sine wave), a small SRT and a deterministic fake clip plan.

Render one clip and assert:

- file exists;
- width = 1080;
- height = 1920;
- duration is expected;
- audio stream exists;
- exit code is 0.

Mock the LLM in CI; tests should not spend API money.

---

# 12. README/documentation updates

The README currently markets the project as Hindi-first and describes the original digital-freedom inspiration. That history can remain, but it should be separated from current product behavior.

Recommended wording:

> Freedom Clips AI started from a Hindi/digital-freedom video workflow but the clip generators are topic- and language-agnostic by default.

Update examples to include:

### Generic single-language

```bash
python ai_opusclip.py --video interview.mp4 --srt interview.srt --out clips
```

### Optional bilingual context

```bash
python ai_opusclip.py \
  --video interview.mp4 \
  --srt transcript_en.srt \
  --secondary-srt transcript_hi.srt \
  --hook-language hi \
  --out clips
```

### Montage

Document explicitly:

> Montage generation uses only the original source video plus local text overlays. It does not call an image generation API.

---

# 13. Cleanup of `noimg.py`

`noimg.py` should not become a second production implementation.

It currently still contains optional image-generation logic and duplicates a large amount of the montage code.

After the real no-image refactor:

- migrate any useful improvements from `noimg.py`;
- delete it, or move it under `experiments/legacy_noimg.py`;
- have one authoritative montage implementation.

This avoids fixing the same bugs twice.

---

# 14. Recommended implementation order

## Phase 1 — P0 / immediate

1. Remove generated image path from `ai_viral_montage_clips.py`.
2. Render hook text over the first original-video beat.
3. Fix A/V speed synchronization.
4. Fix timestamp rounding.
5. Rename/fix `burn_subtitles_and_cta()`.
6. Make `ai_opusclip.py` prompt and schema generic.
7. Add basic preflight validation to both scripts.
8. Update requirements.

## Phase 2 — reliability and cost

9. Add strict JSON schema to montage planning.
10. Add LLM plan caching.
11. Add `--plan-only` and `--render-only`.
12. Improve natural clip-boundary selection.
13. Improve multilingual subtitle alignment.
14. Centralize Pillow text overlays.
15. Add tests.

## Phase 3 — maintainability/performance

16. Extract shared `freedom_clips/` modules.
17. Remove/relocate `noimg.py`.
18. Reduce repeated video re-encoding.
19. Add optional long-transcript chunking/reranking.
20. Update README and changelog for a `0.2.0` release.

---

# 15. Acceptance criteria

The refactor is complete when all of the following are true.

## Generic OpusClip

- A Russian, English, Hindi, Arabic or Chinese SRT can be used without changing source code.
- Default prompt contains no reference to digital freedom.
- Default prompt contains no assumption that the video is Hindi.
- Hook language defaults to the source language.
- JSON keys are language-neutral.
- Output filenames preserve general Unicode safely.

## No generated images

- Running montage generation cannot call an image generation API.
- `gpt-image-1` is absent from the production montage code.
- Hook starts on original footage with original audio.
- No `image_prompt` is required from the LLM.
- No generated image cache/directories are created.

## Correctness

- Video and audio are synchronized when speed != 1.
- Subtitle/overlay timing uses the actual sped-up duration.
- Timestamp formatters cannot emit `,1000` milliseconds or `.100` centiseconds.
- Actual rendered duration is checked with ffprobe.
- Invalid LLM plans fail before rendering.

## Cost

- Re-rendering an unchanged plan does not require a new LLM call.
- No image API spending is possible from the normal clip generators.
- Tests never call paid APIs.

## Maintainability

- Shared SRT and ffmpeg utilities are not duplicated across both main scripts.
- Production behavior exists in one montage implementation, not both `ai_viral_montage_clips.py` and `noimg.py`.
- Automated tests cover core timing and plan validation.

---

# 16. Final recommendation

Do **not** rewrite the whole repository.

The best next release is a focused `0.2.0`:

```text
topic-agnostic + language-agnostic
no image generation
correct A/V speed
plan caching
shared timing/text utilities
basic tests
```

That preserves the strength of the project — a hackable local Python + ffmpeg workflow — while removing the biggest cost trap and the strongest assumptions left over from the original Durov/Hindi use case.

The biggest technical priority is the A/V speed synchronization bug. The biggest product priority is removing subject/language hardcoding. The biggest cost priority is eliminating generated images and making LLM plans reusable.
