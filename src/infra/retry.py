"""Robust call wrappers (§6.6.2): don't pay twice for transient failures, never crash a run.

- `@retry`         : exponential backoff + jitter on a set of retryable exceptions.
- `parse_judge_json`: re-prompt-friendly JSON guard for judge/monitor outputs. After k
                      unparseable replies, returns {"error": "judge_parse"} — callers must
                      NOT cache this as success, so it is retried in a later run.
- `oom_guard`      : on CUDA OOM, halve max_new_tokens/batch once and retry (lazy torch).
"""
from __future__ import annotations

import functools
import json
import random
import re
import time
from typing import Any, Callable, Iterable, Optional


class RetryableError(Exception):
    """Base for explicitly-retryable failures (rate limit, timeout, 5xx)."""


def retry(
    max_attempts: int = 5,
    base: float = 1.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
    sleep: Callable[[float], None] = time.sleep,
):
    """Exponential backoff decorator. `sleep` is injectable so tests run instantly."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except retry_on:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    delay = base * (2 ** (attempt - 1))
                    if jitter:
                        delay *= 0.5 + random.random()
                    sleep(delay)

        return wrapper

    return decorator


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """Best-effort: parse the first {...} block out of an LLM reply."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_judge_json(
    call_fn: Callable[[], str],
    *,
    required_keys: Iterable[str] = (),
    max_attempts: int = 3,
) -> dict:
    """Call an LLM that should return JSON; re-prompt up to k times; on failure return
    {"error": "judge_parse"} (a sentinel the caller must not cache as a success)."""
    required = tuple(required_keys)
    for _ in range(max_attempts):
        raw = call_fn()
        parsed = extract_json(raw)
        if parsed is not None and all(k in parsed for k in required):
            return parsed
    return {"error": "judge_parse"}


def is_parse_error(result: Any) -> bool:
    return isinstance(result, dict) and result.get("error") == "judge_parse"


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def oom_guard(gen_fn: Callable[..., Any], *, max_new_tokens: int, batch_size: int = 1, **kwargs) -> Any:
    """Run gen_fn(max_new_tokens=..., batch_size=..., **kwargs); on CUDA OOM, halve both
    once and retry. Re-raises if it OOMs again. torch is imported lazily by gen_fn."""
    try:
        return gen_fn(max_new_tokens=max_new_tokens, batch_size=batch_size, **kwargs)
    except RuntimeError as exc:
        if not is_oom(exc):
            raise
        return gen_fn(
            max_new_tokens=max(1, max_new_tokens // 2),
            batch_size=max(1, batch_size // 2),
            **kwargs,
        )
