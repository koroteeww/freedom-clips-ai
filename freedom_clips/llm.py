from __future__ import annotations

import json
import os
import re
import time
from typing import Any

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")


def get_openai_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install -r requirements.txt") from exc
    return OpenAI(api_key=api_key)


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    if exc.__class__.__name__ in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }:
        return True
    status = _status_code(exc)
    return status in {408, 409, 429, 500, 502, 503, 504}


def _looks_like_structured_output_error(exc: Exception) -> bool:
    if _status_code(exc) != 400:
        return False
    text = str(exc).lower()
    markers = ("json_schema", "structured output", "response format", "text.format")
    return any(marker in text for marker in markers)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _plain_json_fallback(
    client: Any,
    model: str,
    instructions: str,
    user_input: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    fallback_input = (
        user_input
        + "\n\nReturn valid JSON only. Match this JSON Schema exactly:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=fallback_input,
        max_output_tokens=max_output_tokens,
    )
    return json.loads(_strip_json_fence(response.output_text))


def call_structured_json(
    *,
    client: Any,
    model: str,
    instructions: str,
    user_input: str,
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int = 5000,
    retries: int = 2,
) -> dict[str, Any]:
    """Call the Responses API with strict JSON Schema output.

    Network/rate-limit failures receive bounded exponential retry. A plain-JSON
    fallback is used only when the server rejects structured output or a response
    unexpectedly fails JSON parsing; transport failures do not trigger a second
    fallback request after retries are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
                max_output_tokens=max_output_tokens,
            )
            try:
                return json.loads(response.output_text)
            except json.JSONDecodeError:
                return _plain_json_fallback(
                    client, model, instructions, user_input, schema, max_output_tokens
                )
        except Exception as exc:
            last_error = exc
            if _looks_like_structured_output_error(exc):
                return _plain_json_fallback(
                    client, model, instructions, user_input, schema, max_output_tokens
                )
            if not _is_retryable(exc) or attempt >= retries:
                raise
            time.sleep(min(8.0, 1.5 * (2**attempt)))

    assert last_error is not None
    raise last_error
