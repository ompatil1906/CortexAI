"""Thin, dependency-free LLM client used by ``label_llm.py``.

Supports OpenAI and Gemini behind one interface, with a ``.env`` loader so the
API key does not have to be exported into the caller's shell.

Resolution order for a key:
    1. the process environment
    2. ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` or ``OPENAI_API_KEY`` from ``.env``
    3. ``--api-key-file``

Only ``urllib`` is used, so this adds no install-time dependency.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_vars": ("OPENAI_API_KEY",),
        "default_model": "gpt-4.1-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "key_vars": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "default_model": "gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
    },
}


def load_dotenv(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a minimal ``KEY=value`` .env into the process env without clobbering it."""
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def resolve_key(provider: str, api_key_file: str | None = None) -> str:
    load_dotenv()
    if api_key_file:
        key = Path(api_key_file).read_text().strip()
        if key:
            return key

    config = PROVIDERS[provider]
    for name in config["key_vars"]:
        value = os.environ.get(name)
        if value:
            return value

    checked = " or ".join(config["key_vars"])
    raise SystemExit(
        f"No API key found for provider {provider!r}.\n"
        f"Looked in the environment and in {ENV_FILE} for: {checked}\n"
        f"Fix either by exporting the variable, adding it to {ENV_FILE}, "
        f"or passing --api-key-file <path>."
    )


def infer_provider(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith(("gemini", "models/gemini")):
        return "gemini"
    if lowered.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    raise SystemExit(
        f"Cannot infer a provider from model {model!r}. "
        f"Pass --provider explicitly (one of: {', '.join(PROVIDERS)})."
    )


def build_request(
    provider: str, model: str, system_prompt: str, user_prompt: str, schema: dict, seed: int
) -> tuple[str, dict]:
    """Return (url, request body) for a structured-output chat call."""
    if provider == "openai":
        url = PROVIDERS["openai"]["url"]
        body = {
            "model": model,
            "temperature": 0,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema["name"], "strict": True, "schema": schema["schema"]},
            },
        }
        return url, body

    # Gemini: system instruction is a separate field, and structured output is
    # requested via responseSchema + responseMimeType.
    url = PROVIDERS["gemini"]["url"].format(model=model)
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "seed": seed,
            "responseMimeType": "application/json",
            "responseSchema": sanitize_schema_for_gemini(schema["schema"]),
        },
    }
    return url, body


def sanitize_schema_for_gemini(schema: dict) -> dict:
    """Strip JSON-Schema keywords Gemini's ``responseSchema`` does not accept.

    Gemini supports a subset of JSON Schema and rejects the rest outright with
    ``Unknown name "additionalProperties" ... Cannot find field`` (HTTP 400). That
    keyword is an OpenAI strict-mode requirement, so the same schema object cannot
    be sent to both providers unmodified. Only the unsupported keys are removed;
    ``type``/``properties``/``required``/``enum``/``items`` pass through untouched,
    which is what actually constrains the output.
    """
    unsupported = {"additionalProperties", "$schema", "$id", "strict", "definitions", "$defs"}
    if isinstance(schema, dict):
        return {
            key: sanitize_schema_for_gemini(value)
            for key, value in schema.items()
            if key not in unsupported
        }
    if isinstance(schema, list):
        return [sanitize_schema_for_gemini(item) for item in schema]
    return schema


def build_headers(provider: str, api_key: str, model: str) -> dict[str, str]:
    if provider == "openai":
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def parse_response(provider: str, body: dict) -> tuple[dict | None, dict | None]:
    """Return (parsed_json, usage) or (None, None) when the shape is unexpected."""
    usage = body.get("usage") or body.get("usageMetadata") or {}

    if provider == "openai":
        try:
            content = body["choices"][0]["message"]["content"]
            return json.loads(content), usage
        except Exception:
            return None, None

    try:
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
        return json.loads(text), usage
    except Exception:
        return None, None


def post_json(
    url: str, headers: dict, body: dict, timeout: int = 120
) -> tuple[int, dict] | tuple[int, None]:
    """POST JSON, returning (status_code, parsed_body_or_None). HTTP errors raise."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Return the error body rather than None. Providers explain themselves in it
        # ("Unsupported property: additionalProperties"), and discarding it turns a
        # one-minute fix into blind guesswork -- every failure looks like "HTTP 400".
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"error": "unparseable error body"}
        return exc.code, detail


def is_retryable(status: int) -> bool:
    return status == 429 or status >= 500


def backoff(attempt: int, cap: float = 60.0) -> float:
    return min(2**attempt + random.random(), cap)


__all__ = [
    "PROVIDERS",
    "build_headers",
    "build_request",
    "infer_provider",
    "is_retryable",
    "load_dotenv",
    "parse_response",
    "post_json",
    "resolve_key",
    "backoff",
]