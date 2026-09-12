"""The one place Cry Wolf talks to a model.

`observer.py` asks for a validated Pydantic object and does not know or care how
it was produced. Everything provider-shaped lives here.

    ANTHROPIC_API_KEY=...          required
    CRYWOLF_MODEL=claude-haiku-4-5 default; raise it when reads look shallow
    CRYWOLF_EFFORT=medium          ignored on models that don't support effort
    CRYWOLF_MIN_INTERVAL=0         seconds between calls, if you get rate-limited

Output is constrained by the schema via `messages.parse`, so nothing here ever
parses prose -- a malformed response raises rather than quietly degrading.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Protocol, Type, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 6

# Start cheap and fast. One replay is ~27 calls and tuning means many replays,
# so the ladder is haiku -> claude-sonnet-5 -> claude-opus-5 as quality demands.
DEFAULT_MODEL = "claude-haiku-4-5"

# `output_config.effort` is rejected by Haiku 4.5 and Sonnet 4.5 -- sending it
# there is a 400 on every call, not a warning. Newer models accept it.
_NO_EFFORT = re.compile(r"haiku|sonnet-4-5", re.I)

# A 429 body states how long to wait ("retry in 52.14s"). Believe it over a guess.
_RETRY_HINT = re.compile(r"retry(?:-|\s|_)?(?:after|delay)?['\"]?[:\s]+['\"]?(\d+(?:\.\d+)?)s?", re.I)


class StructuredLLM(Protocol):
    """The entire contract. Anything with this method can drive the observer."""

    def structured(self, system: str, user: str, schema: Type[T]) -> T: ...


class _Throttle:
    """Keeps a minimum gap between calls. Zero by default -- set
    CRYWOLF_MIN_INTERVAL if a low rate limit starts biting."""

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


def _suggested_delay(text: str) -> float | None:
    match = _RETRY_HINT.search(text)
    return float(match.group(1)) if match else None


def _retry(call, label: str = "claude"):
    """Retry rate limits and transient server errors, honoring the server's own
    requested delay when it states one."""
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            status = getattr(exc, "status_code", None)
            retryable = isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)) or (
                status is not None and status >= 500
            )
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            hinted = _suggested_delay(str(exc))
            # A second of slack -- retrying at the exact boundary often 429s again.
            delay = (hinted + 1.0) if hinted else (2**attempt) + random.random()
            why = "server asked" if hinted else "backoff"
            print(f"    [{label} throttled, waiting {delay:.0f}s ({why})]", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


class ClaudeBackend:
    """Claude via the Messages API, with schema-constrained output."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        if not (api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "No Anthropic key found. Set ANTHROPIC_API_KEY "
                "(console.anthropic.com/settings/keys)."
            )
        self.model = model or os.environ.get("CRYWOLF_MODEL") or DEFAULT_MODEL
        self.effort = os.environ.get("CRYWOLF_EFFORT", "medium")
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._throttle = _Throttle(float(os.environ.get("CRYWOLF_MIN_INTERVAL", 0)))

    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        extra = {}
        if not _NO_EFFORT.search(self.model):
            extra["output_config"] = {"effort": self.effort}

        def call():
            self._throttle.wait()
            return self._client.messages.parse(
                model=self.model,
                max_tokens=8000,
                # The system prompt is identical on every event, so cache it --
                # 27 calls a run means 26 cache reads instead of 26 re-sends.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                **extra,
            )

        return _retry(call).parsed_output


def get_backend() -> StructuredLLM:
    return ClaudeBackend()
