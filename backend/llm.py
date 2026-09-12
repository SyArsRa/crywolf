"""The one place Cry Wolf talks to a model.

`observer.py` asks for a validated Pydantic object and does not know or care
which provider produced it. Swapping providers is an env var, not an edit:

    CRYWOLF_PROVIDER=gemini     (default)  needs GEMINI_API_KEY
    CRYWOLF_PROVIDER=anthropic             needs ANTHROPIC_API_KEY
    CRYWOLF_MODEL=...                      override the per-provider default

Both backends use native structured output -- the model is constrained to the
schema rather than asked nicely for JSON -- so nothing here ever parses prose.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Protocol, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 6

# Free-tier Gemini is 5 requests/minute, and one replay is ~27 calls back to
# back -- so we pace ourselves rather than sprinting into a 429 every sixth
# event. 13s keeps us just under the limit with room for clock skew.
DEFAULT_MIN_INTERVAL = {"gemini": 13.0, "anthropic": 0.0}

# The server tells us how long to wait; a 429 body carries either
# "Please retry in 52.14s" or "retryDelay: '52s'". Believe it over our own guess.
_RETRY_HINT = re.compile(r"retry(?:delay)?['\"]?[:\s]+['\"]?(\d+(?:\.\d+)?)s", re.I)


class StructuredLLM(Protocol):
    """The entire contract. Anything with this method can drive the observer."""

    def structured(self, system: str, user: str, schema: Type[T]) -> T: ...


class _Throttle:
    """Keeps a minimum gap between calls, so we approach the quota instead of
    slamming into it. Free tiers punish bursts far more than steady pacing."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


def _min_interval(provider: str) -> float:
    """Seconds between calls. Override with CRYWOLF_MIN_INTERVAL -- set it to 0
    on a paid key, or raise it if you're still getting throttled."""
    override = os.environ.get("CRYWOLF_MIN_INTERVAL")
    if override is not None:
        return float(override)
    return DEFAULT_MIN_INTERVAL.get(provider, 0.0)


def _suggested_delay(text: str) -> float | None:
    match = _RETRY_HINT.search(text)
    return float(match.group(1)) if match else None


def _retry(call, label: str):
    """Retry rate limits and transient server errors, honoring the server's
    own requested delay when it states one."""
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except Exception as exc:  # provider SDKs raise unrelated exception trees
            text = str(exc)
            low = text.lower()
            transient = (
                "429" in low
                or "rate" in low
                or "quota" in low
                or "resource_exhausted" in low
                or "unavailable" in low
                or "500" in low
                or "503" in low
                or "overloaded" in low
            )
            if not transient or attempt == MAX_RETRIES - 1:
                raise
            hinted = _suggested_delay(text)
            # Add a second of slack -- retrying at the exact boundary often 429s again.
            delay = (hinted + 1.0) if hinted else (2**attempt) + random.random()
            why = "server asked" if hinted else "backoff"
            print(f"    [{label} throttled, waiting {delay:.0f}s ({why})]", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


class GeminiBackend:
    """Google Gemini via google-genai.

    Defaults to flash rather than pro: a tuning session means replaying the game
    many times, and the free tier's pro quota does not survive that.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini key found. Set GEMINI_API_KEY (get one free at "
                "https://aistudio.google.com/apikey), or set CRYWOLF_PROVIDER=anthropic."
            )
        self.model = model or os.environ.get("CRYWOLF_MODEL") or self.DEFAULT_MODEL
        self._client = genai.Client(api_key=key)
        self._throttle = _Throttle(_min_interval("gemini"))
        # We pass a response_schema, never tools; the SDK's automatic-function-calling
        # advice doesn't apply to us and only clutters the demo output.
        logging.getLogger("google_genai.models").setLevel(logging.ERROR)

    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        from google.genai import types

        def call():
            self._throttle.wait()
            return self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )

        response = _retry(call, "gemini")
        parsed = response.parsed
        if parsed is None:
            raise RuntimeError(f"Gemini returned no parsable object: {response.text!r}")
        return parsed


class AnthropicBackend:
    """Claude via the Messages API. Kept working for whenever a key turns up."""

    DEFAULT_MODEL = "claude-opus-5"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic

        self.model = model or os.environ.get("CRYWOLF_MODEL") or self.DEFAULT_MODEL
        self.effort = os.environ.get("CRYWOLF_EFFORT", "medium")
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._throttle = _Throttle(_min_interval("anthropic"))

    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        def call():
            self._throttle.wait()
            return self._client.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": self.effort},
            )

        return _retry(call, "anthropic").parsed_output


def get_backend() -> StructuredLLM:
    provider = os.environ.get("CRYWOLF_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return GeminiBackend()
    if provider == "anthropic":
        return AnthropicBackend()
    raise RuntimeError(f"Unknown CRYWOLF_PROVIDER {provider!r} (expected 'gemini' or 'anthropic')")
