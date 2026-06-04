"""Reference-tokenizer length enforcement (§5.5).

Key design decisions:
  - Length is measured ONCE in the `reference_tokenizer` (default: DeepSeek-R1-Distill-1.5B).
    The SAME prefill is reused across all subject models (huge saving for the size sweep).
  - Per-subject token counts are recorded as metadata (so cross-model length drift is visible).
  - Tolerance: ±12% of the target bucket (e.g., target=300 → [264, 336]).
  - Too short  → ask the generator to expand (call generator_fn again with a larger target).
  - Too long   → truncate at the last sentence boundary BEFORE the conclusion, then
                 re-append the conclusion sentence.
  - Retry up to `max_retries` times; failures are logged (never silently kept).
  - A `MockTokenizer` counts whitespace-split words×1.3 so tests run offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Tokenizer utilities ───────────────────────────────────────────────────────

class MockTokenizer:
    """Offline stand-in: 1 token ≈ 0.75 words (word count × 1.33)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.split()
        n = max(1, round(len(words) * 1.33))
        return list(range(n))  # fake token ids, only length matters


def token_count(text: str, tokenizer: Any) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def load_reference_tokenizer(model_id: str):
    """Load the reference tokenizer from HuggingFace (lightweight, no GPU needed)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def target_range(target_tokens: int, tolerance: float = 0.12) -> tuple[int, int]:
    lo = max(1, round(target_tokens * (1 - tolerance)))
    hi = round(target_tokens * (1 + tolerance))
    return lo, hi


def in_range(text: str, target_tokens: int, tokenizer: Any, tolerance: float = 0.12) -> bool:
    lo, hi = target_range(target_tokens, tolerance)
    n = token_count(text, tokenizer)
    return lo <= n <= hi


# ── Truncation ────────────────────────────────────────────────────────────────

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def truncate_to_length(
    text: str,
    target_tokens: int,
    conclusion: str,
    tokenizer: Any,
    tolerance: float = 0.12,
) -> str:
    """Truncate `text` to fit within target_tokens ± tolerance, then re-append conclusion.

    Strategy: split into sentences, drop from the end until the body + conclusion fits,
    then re-attach the conclusion. If the text is already short enough, return as-is.
    """
    lo, hi = target_range(target_tokens, tolerance)

    # Strip any existing conclusion from the end before re-appending.
    body = text
    if body.rstrip().endswith(conclusion.rstrip(".")):
        # Remove the conclusion sentence.
        idx = body.rfind(conclusion[:30])
        if idx > 0:
            body = body[:idx].rstrip()

    full = body.rstrip() + "\n" + conclusion
    if token_count(full, tokenizer) <= hi:
        return full  # already fits; may be under-length (caller decides)

    # Try sentence-level truncation first.
    sentences = _SENT_BOUNDARY.split(body)
    while len(sentences) > 1:
        sentences.pop()
        candidate = " ".join(sentences).rstrip() + "\n" + conclusion
        if token_count(candidate, tokenizer) <= hi:
            return candidate

    # Fall back to word-level truncation (handles texts with no sentence boundaries).
    words = body.split()
    while len(words) > 1:
        words.pop()
        candidate = " ".join(words) + "\n" + conclusion
        if token_count(candidate, tokenizer) <= hi:
            return candidate

    # Can't truncate further; return minimal form.
    return conclusion


# ── Length enforcement loop ───────────────────────────────────────────────────

@dataclass
class LengthControlResult:
    text: str
    token_count: int
    target_tokens: int
    in_range: bool
    attempts: int
    failure_reason: Optional[str] = None
    per_model_counts: dict[str, int] = field(default_factory=dict)


def enforce_length(
    initial_text: str,
    target_tokens: int,
    conclusion: str,
    tokenizer: Any,
    *,
    generator_fn: Optional[Callable[[int], str]] = None,
    max_retries: int = 3,
    tolerance: float = 0.12,
    extra_tokenizers: Optional[dict[str, Any]] = None,
) -> LengthControlResult:
    """Enforce that `text` is within target_tokens ± tolerance in `tokenizer`.

    - If too long: truncate at sentence boundary, re-append conclusion.
    - If too short: call generator_fn(new_target) to regenerate (if provided).
    - Retries up to max_retries times before logging failure.
    - Records per-model token counts from `extra_tokenizers` as metadata.
    """
    lo, hi = target_range(target_tokens, tolerance)
    text = initial_text
    attempts = 0
    failure_reason: Optional[str] = None

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        n = token_count(text, tokenizer)

        if lo <= n <= hi:
            # In range — record cross-model counts and return.
            per_model = {}
            if extra_tokenizers:
                for name, tok in extra_tokenizers.items():
                    per_model[name] = token_count(text, tok)
            return LengthControlResult(
                text=text,
                token_count=n,
                target_tokens=target_tokens,
                in_range=True,
                attempts=attempts,
                per_model_counts=per_model,
            )

        if n > hi:
            # Too long: truncate.
            text = truncate_to_length(text, target_tokens, conclusion, tokenizer, tolerance)
            # Truncation is deterministic; one pass is enough — don't retry truncation.
            n2 = token_count(text, tokenizer)
            if lo <= n2 <= hi:
                per_model = {}
                if extra_tokenizers:
                    for name, tok in extra_tokenizers.items():
                        per_model[name] = token_count(text, tok)
                return LengthControlResult(
                    text=text, token_count=n2, target_tokens=target_tokens,
                    in_range=True, attempts=attempts, per_model_counts=per_model,
                )
            # Truncation produced something still out of range (rare).
            failure_reason = f"truncation landed at {n2} tokens (target [{lo},{hi}])"
            break

        # Too short: ask generator to expand (slightly above target to have room to trim).
        if generator_fn is None:
            failure_reason = f"text has {n} tokens < lo={lo}, no generator_fn provided"
            break

        new_target = round(target_tokens * 1.15)
        try:
            text = generator_fn(new_target)
        except Exception as e:
            failure_reason = f"generator_fn raised on attempt {attempt+1}: {e}"
            break

    else:
        n = token_count(text, tokenizer)
        if not (lo <= n <= hi):
            failure_reason = f"max_retries={max_retries} exceeded; final count={n}, target=[{lo},{hi}]"

    # Log failure (callers can also inspect the returned object).
    if failure_reason:
        import warnings
        warnings.warn(f"[length_control] {failure_reason}")

    n_final = token_count(text, tokenizer)
    per_model = {}
    if extra_tokenizers:
        for name, tok in extra_tokenizers.items():
            per_model[name] = token_count(text, tok)

    return LengthControlResult(
        text=text,
        token_count=n_final,
        target_tokens=target_tokens,
        in_range=(lo <= n_final <= hi),
        attempts=attempts,
        failure_reason=failure_reason,
        per_model_counts=per_model,
    )
