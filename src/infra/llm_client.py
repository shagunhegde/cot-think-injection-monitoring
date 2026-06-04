"""Thin LLM client for judges, monitors, and attack generators.

Usage:
    client = LLMClient.from_env()          # auto-selects a backend (see below)
    reply = client.chat(model, messages, temperature=0)

Messages follow the OpenAI-style list format (system role supported). Each backend
adapts that uniform interface to its provider so callers never branch on provider.

Backends:
  - OpenRouterBackend  — DEFAULT. One key (OPENROUTER_API_KEY) reaches any open or
                         closed model (anthropic/…, openai/…, qwen/…, google/…). This is
                         what makes the attacker + monitor model SWEEP uniform: every
                         model is just a different `model` string on the same client.
  - AnthropicBackend   — direct Anthropic Messages API (ANTHROPIC_API_KEY).
  - MockBackend        — offline, no key; used by the test suite.

Backend selection (`from_env`):
  COTIM_LLM_BACKEND=mock|openrouter|anthropic forces a backend; otherwise the first
  key present wins (OPENROUTER_API_KEY → ANTHROPIC_API_KEY).

Cost knobs (price_per_1k, avg_tokens) come from config/cost.yaml; this module only
handles call transport.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from .retry import RetryableError, retry


class LLMBackend(ABC):
    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """Return the assistant reply text."""


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API backend."""

    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic SDK required: pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic = anthropic

    @retry(max_attempts=5, base=1.0, jitter=True, retry_on=(RetryableError,))
    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 1024, **kwargs: Any) -> str:
        # Anthropic separates system prompt from the messages list.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") != "system"]

        call_kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=user_messages,
            **kwargs,
        )
        if system_parts:
            call_kwargs["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        try:
            resp = self._client.messages.create(**call_kwargs)
            return resp.content[0].text if resp.content else ""
        except self._anthropic.RateLimitError as e:
            raise RetryableError(str(e)) from e
        except self._anthropic.APITimeoutError as e:
            raise RetryableError(str(e)) from e
        except self._anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise RetryableError(str(e)) from e
            raise


class OpenRouterBackend(LLMBackend):
    """OpenRouter backend (OpenAI-compatible Chat Completions).

    A single OPENROUTER_API_KEY reaches any model on OpenRouter, so the attacker and
    the weak→strong monitor sweep are all just different `model` strings on one client.
    The OpenAI-style messages list (incl. system role) is passed through unchanged.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, *, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK required for OpenRouter: pip install 'openai>=1.0' "
                "(or install the '[openrouter]' extra)."
            ) from e
        # OpenRouter recommends these headers for attribution; harmless if omitted.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or self.BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/cot-injection-monitoring",
                "X-Title": "cot-injection-monitoring",
            },
        )
        from openai import APIError, APITimeoutError, RateLimitError
        self._APIError = APIError
        self._APITimeoutError = APITimeoutError
        self._RateLimitError = RateLimitError

    @retry(max_attempts=5, base=1.0, jitter=True, retry_on=(RetryableError,))
    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 1024, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,          # system role passes through directly
            max_tokens=max_tokens,
            **kwargs,
        )
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        try:
            resp = self._client.chat.completions.create(**call_kwargs)
            return resp.choices[0].message.content or "" if resp.choices else ""
        except self._RateLimitError as e:
            raise RetryableError(str(e)) from e
        except self._APITimeoutError as e:
            raise RetryableError(str(e)) from e
        except self._APIError as e:
            # Retry transient upstream failures (5xx / no status); surface client errors.
            status = getattr(e, "status_code", None)
            if status is None or status >= 500:
                raise RetryableError(str(e)) from e
            raise


class MockBackend(LLMBackend):
    """Offline mock for tests. Returns canned JSON or a configured response."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self._calls: list[dict] = []
        self._idx = 0

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 1024, **kwargs: Any) -> str:
        self._calls.append({"model": model, "messages": messages})
        if self._responses:
            reply = self._responses[self._idx % len(self._responses)]
            self._idx += 1
            return reply
        # Default: echo the last user message in a JSON envelope (useful for judges).
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return json.dumps({"flag": False, "rationale": f"mock: {last_user[:80]}"})

    @property
    def calls(self) -> list[dict]:
        return self._calls


class LLMClient:
    """Public API: wrap a backend and expose `.chat(...)`."""

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 1024, **kwargs: Any) -> str:
        return self._backend.chat(model, messages, temperature=temperature,
                                  max_tokens=max_tokens, **kwargs)

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Auto-select a backend from the environment.

        Resolution order:
          1. COTIM_LLM_BACKEND=mock|openrouter|anthropic forces that backend.
          2. Otherwise, the first key present wins: OPENROUTER_API_KEY → ANTHROPIC_API_KEY.
        """
        forced = os.environ.get("COTIM_LLM_BACKEND", "").lower()
        if forced == "mock":
            return cls(MockBackend())
        if forced == "openrouter":
            return cls(OpenRouterBackend(_require_key("OPENROUTER_API_KEY")))
        if forced == "anthropic":
            return cls(AnthropicBackend(_require_key("ANTHROPIC_API_KEY")))

        if os.environ.get("OPENROUTER_API_KEY"):
            return cls(OpenRouterBackend(os.environ["OPENROUTER_API_KEY"]))
        if os.environ.get("ANTHROPIC_API_KEY"):
            return cls(AnthropicBackend(os.environ["ANTHROPIC_API_KEY"]))

        raise EnvironmentError(
            "No LLM key found. Set OPENROUTER_API_KEY (recommended) or ANTHROPIC_API_KEY, "
            "or COTIM_LLM_BACKEND=mock for offline testing."
        )


def _require_key(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise EnvironmentError(f"{name} not set (required by COTIM_LLM_BACKEND).")
    return val
