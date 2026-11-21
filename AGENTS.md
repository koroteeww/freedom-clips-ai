# AGENTS.md — Freedom Clips AI

This file is guidance for coding agents and automated contributors.

## Project goal

Keep Freedom Clips AI a small, local-first Python + ffmpeg toolkit for turning SRT-aligned long-form video into vertical clips.

## Non-negotiable invariants

1. Do not add generated-art API calls to `ai_viral_montage_clips.py` or its production modules.
2. Do not hardcode a content topic, ideology, country, or source language into default prompts.
3. Keep `--render-only --plan ...` usable without an OpenAI API key.
4. If video speed changes, transform video PTS and audio tempo consistently.
5. Do not silently cut continuous clips at arbitrary timestamps when natural SRT boundaries are available.
6. Tests must not call paid APIs.
7. Keep shared behavior in `freedom_clips/`; top-level scripts should remain orchestration/CLI layers.

## Before changing timing or ffmpeg code

Run:

```bash
pytest -q
```

If ffmpeg is installed, do not skip the A/V integration test when validating speed-related changes.

## Prompt changes

When editing default prompts:

- preserve subject neutrality;
- preserve language neutrality;
- preserve factual/integrity constraints;
- update the prompt version constant so cached plans are invalidated intentionally;
- run `tests/test_prompts.py`.

## Plan schema changes

If the JSON schema changes:

- update the corresponding validator;
- update README output examples;
- update sample plans under `examples/`;
- consider backward compatibility for existing plan files;
- add or update tests.

## Dependency philosophy

Prefer the standard library plus:

- `openai`
- `Pillow`
- external `ffmpeg` / `ffprobe`

Do not add a large video framework unless it removes significant complexity and remains optional.

## Cost philosophy

Any new paid API call requires a clear justification, explicit user action, and a cache/reuse strategy. Local computation is preferred for rendering.
