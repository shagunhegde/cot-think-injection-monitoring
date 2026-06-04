"""Attack prefill generation — cache Layer 2 (§6.6.1).

Generates and caches prefills for Families A (template), B, C, D (Anthropic API generator).
Prefills are MODEL-AGNOSTIC: the same content is reused across all subject models
(length measured in the reference tokenizer once; per-model counts stored as metadata).

C prefills are BUILT PER-B: for each B prefill of measured length L, a C prefill of
length ≈L is generated and validated. This is the matched-pairs requirement for CAS.

Shard scheme for Layer 2:
    prefills/{dataset}_{family}_{length_bucket}.jsonl

The cache key includes (dataset, item_hash, family, length_bucket, target_letter,
generator_model, prompt_version) so a prompt tweak only invalidates Layer 2 for that
version; Layer 3 GPU generations are untouched.

Validation failures are logged and NOT cached. The caller can re-run to retry them.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from src.infra.cache import Cache, canonical_key
from src.infra.llm_client import LLMClient
from src.infra.retry import parse_judge_json, is_parse_error
from .templates import (
    build_family_a,
    build_generator_prompt,
    conclusion_sentence,
)
from .length_control import (
    MockTokenizer,
    enforce_length,
    token_count,
    LengthControlResult,
)
from .validate import validate_prefill

log = logging.getLogger(__name__)


# ── Cache key ─────────────────────────────────────────────────────────────────

def prefill_cache_key(
    dataset: str,
    item_hash: str,
    family: str,
    length_bucket: int,
    target_letter: str,
    generator_model: str,
    prompt_version: str,
) -> str:
    return canonical_key(
        dataset=dataset,
        item_hash=item_hash,
        family=family.upper(),
        length_bucket=length_bucket,
        target_letter=target_letter.upper(),
        generator_model=generator_model,
        prompt_version=prompt_version,
    )


def prefill_shard(dataset: str, family: str, length_bucket: int) -> str:
    return f"prefills/{dataset}_{family.upper()}_{length_bucket}"


# ── Generation result ─────────────────────────────────────────────────────────

@dataclass
class PrefillResult:
    text: str
    family: str
    length_bucket: int
    token_count: int
    target_letter: str
    valid: bool
    validation_notes: list[str] = field(default_factory=list)
    length_attempts: int = 1
    per_model_counts: dict[str, int] = field(default_factory=dict)


# ── Family A (template, free) ─────────────────────────────────────────────────

def generate_family_a(
    item: dict,
    target_letter: str,
    length_bucket: int,
    tokenizer: Any,
    *,
    tolerance: float = 0.12,
    dataset: str = "mmlu",
    generator_model: str = "template",
    prompt_version: str = "v1",
    cache: Optional[Cache] = None,
    extra_tokenizers: Optional[dict[str, Any]] = None,
) -> PrefillResult:
    """Build a Family-A prefill (template, no API call)."""
    key = prefill_cache_key(
        dataset, item["item_hash"], "A", length_bucket,
        target_letter, generator_model, prompt_version,
    )
    shard = prefill_shard(dataset, "A", length_bucket)

    def _compute():
        text = build_family_a(target_letter, length_bucket, tokenizer)
        result = enforce_length(
            text, length_bucket, conclusion_sentence(target_letter),
            tokenizer, tolerance=tolerance, extra_tokenizers=extra_tokenizers,
        )
        vr = validate_prefill(
            result.text, "A", item, target_letter,
            target_tokens=length_bucket, tokenizer=tokenizer, tolerance=tolerance,
        )
        return {
            "text": result.text,
            "token_count": result.token_count,
            "valid": vr.passed,
            "validation_notes": vr.notes,
            "length_attempts": result.attempts,
            "per_model_counts": result.per_model_counts,
        }

    if cache is not None:
        cached = cache.get_or_compute(shard=shard, key=key, compute_fn=_compute,
                                      meta={"family": "A", "length_bucket": length_bucket})
    else:
        cached = _compute()

    return PrefillResult(
        text=cached["text"], family="A", length_bucket=length_bucket,
        token_count=cached["token_count"], target_letter=target_letter,
        valid=cached["valid"], validation_notes=cached.get("validation_notes", []),
        length_attempts=cached.get("length_attempts", 1),
        per_model_counts=cached.get("per_model_counts", {}),
    )


# ── Families B, C, D (Anthropic API generator) ───────────────────────────────

def _call_generator(
    client: LLMClient,
    messages: list[dict],
    generator_model: str,
    max_tokens: int = 2048,
) -> str:
    """Call the generator and return the raw text reply."""
    return client.chat(
        generator_model,
        messages,
        temperature=0.7,
        max_tokens=max_tokens,
    )


def generate_prefill(
    item: dict,
    family: str,
    target_letter: str,
    length_bucket: int,
    client: LLMClient,
    tokenizer: Any,
    *,
    tolerance: float = 0.12,
    max_retries: int = 3,
    dataset: str = "mmlu",
    generator_model: str = "claude-haiku-4-5-20251001",
    prompt_version: str = "v1",
    cache: Optional[Cache] = None,
    extra_tokenizers: Optional[dict[str, Any]] = None,
    # For Family C: paired B details.
    paired_b_text: Optional[str] = None,
    paired_b_token_count: Optional[int] = None,
) -> PrefillResult:
    """Generate (or retrieve from cache) a single prefill.

    Validation failures are NOT cached — they will be retried on the next run.
    """
    family = family.upper()
    key = prefill_cache_key(
        dataset, item["item_hash"], family, length_bucket,
        target_letter, generator_model, prompt_version,
    )
    shard = prefill_shard(dataset, family, length_bucket)

    # Check cache first (outside compute_fn so we can return early).
    if cache is not None and cache.contains(shard, key):
        cached = cache.get(shard, key)
        if cached and cached.get("valid", False):
            return PrefillResult(
                text=cached["text"], family=family, length_bucket=length_bucket,
                token_count=cached["token_count"], target_letter=target_letter,
                valid=True, validation_notes=cached.get("validation_notes", []),
                length_attempts=cached.get("length_attempts", 1),
                per_model_counts=cached.get("per_model_counts", {}),
            )

    conclusion = conclusion_sentence(target_letter)

    def _generate_raw(requested_tokens: int) -> str:
        messages, _ = build_generator_prompt(
            item, family, target_letter, requested_tokens,
            paired_b_length=paired_b_token_count,
            prompt_version=prompt_version,
        )
        return _call_generator(client, messages, generator_model)

    # Initial generation.
    raw = _generate_raw(length_bucket)

    lc = enforce_length(
        raw, length_bucket, conclusion, tokenizer,
        generator_fn=_generate_raw,
        max_retries=max_retries,
        tolerance=tolerance,
        extra_tokenizers=extra_tokenizers,
    )

    vr = validate_prefill(
        lc.text, family, item, target_letter,
        target_tokens=length_bucket,
        tokenizer=tokenizer,
        tolerance=tolerance,
        paired_b_text=paired_b_text,
        paired_b_token_count=paired_b_token_count,
    )

    if not vr.passed:
        log.warning(
            "[generate] validation FAIL: family=%s item=%s target=%s bucket=%d — %s",
            family, item["item_hash"][:8], target_letter, length_bucket, vr.notes,
        )
        # Return the result but do NOT cache it (caller must not cache failed prefills).
        return PrefillResult(
            text=lc.text, family=family, length_bucket=length_bucket,
            token_count=lc.token_count, target_letter=target_letter,
            valid=False, validation_notes=vr.notes,
            length_attempts=lc.attempts,
            per_model_counts=lc.per_model_counts,
        )

    record = {
        "text": lc.text,
        "token_count": lc.token_count,
        "valid": True,
        "validation_notes": [],
        "length_attempts": lc.attempts,
        "per_model_counts": lc.per_model_counts,
    }

    if cache is not None:
        # Write immediately (don't use get_or_compute since we already generated).
        from src.infra.cache import canonical_key as _ck
        cache.get_or_compute(shard=shard, key=key, compute_fn=lambda: record,
                             meta={"family": family, "length": length_bucket,
                                   "item_hash": item["item_hash"]})

    return PrefillResult(
        text=record["text"], family=family, length_bucket=length_bucket,
        token_count=record["token_count"], target_letter=target_letter,
        valid=True, per_model_counts=record["per_model_counts"],
    )


# ── Batch prefill generation for a list of conditions ─────────────────────────

@dataclass
class PrefillBatchStats:
    total: int = 0
    cached: int = 0
    generated: int = 0
    failed: int = 0
    failures: list[dict] = field(default_factory=list)


def generate_all_prefills(
    conditions: list[dict],
    items_by_hash: dict[str, dict],
    client: LLMClient,
    tokenizer: Any,
    cache: Cache,
    *,
    tolerance: float = 0.12,
    max_retries: int = 3,
    dataset: str = "mmlu",
    generator_model: str = "claude-haiku-4-5-20251001",
    prompt_version: str = "v1",
    extra_tokenizers: Optional[dict[str, Any]] = None,
) -> PrefillBatchStats:
    """Generate prefills for all conditions. C prefills are automatically matched to B.

    Returns stats (total/cached/generated/failed).
    Failures are logged and NOT cached — they will be retried on re-run.
    """
    stats = PrefillBatchStats()

    # Deduplicate by (item_hash, family, length_bucket, target_letter).
    seen: set[tuple] = set()
    # Pre-collect B results so C can reference them.
    b_results: dict[tuple, PrefillResult] = {}

    # Two-pass: first generate B (and A/D), then C (needs paired B).
    for family_pass in (("A", "B", "D"), ("C",)):
        for cond in conditions:
            family = cond["family"].upper()
            if family not in family_pass:
                continue

            item_hash = cond["item"]
            item = items_by_hash.get(item_hash)
            if item is None:
                continue

            target = cond["target"]
            if target in ("most_plausible", "wrong_0", "wrong_1", "wrong_2"):
                # Target not yet resolved to a letter (needs baseline layer).
                # Skip — caller should resolve targets before calling this.
                continue

            length_bucket = cond["length"]
            dedup_key = (item_hash, family, length_bucket, target)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            stats.total += 1

            # Check cache before generating.
            # Family A always uses "template" as generator_model in its cache key.
            _gen_model_for_key = "template" if family == "A" else generator_model
            ck = prefill_cache_key(
                dataset, item_hash, family, length_bucket,
                target, _gen_model_for_key, prompt_version,
            )
            shard = prefill_shard(dataset, family, length_bucket)
            if cache.contains(shard, ck):
                cached_val = cache.get(shard, ck)
                if cached_val and cached_val.get("valid", False):
                    stats.cached += 1
                    if family == "B":
                        b_results[(item_hash, length_bucket, target)] = PrefillResult(
                            text=cached_val["text"], family="B",
                            length_bucket=length_bucket,
                            token_count=cached_val["token_count"],
                            target_letter=target, valid=True,
                            per_model_counts=cached_val.get("per_model_counts", {}),
                        )
                    continue

            # Generate.
            paired_b: Optional[PrefillResult] = None
            if family == "C":
                paired_b = b_results.get((item_hash, length_bucket, target))
                if paired_b is None:
                    log.warning(
                        "[generate_all] Family C skipped: no paired B for "
                        "item=%s length=%d target=%s", item_hash[:8], length_bucket, target,
                    )
                    stats.failed += 1
                    continue

            if family == "A":
                result = generate_family_a(
                    item, target, length_bucket, tokenizer,
                    tolerance=tolerance, dataset=dataset,
                    generator_model="template",  # A is always template-based
                    prompt_version=prompt_version,
                    cache=cache, extra_tokenizers=extra_tokenizers,
                )
            else:
                result = generate_prefill(
                    item, family, target, length_bucket,
                    client, tokenizer,
                    tolerance=tolerance, max_retries=max_retries,
                    dataset=dataset, generator_model=generator_model,
                    prompt_version=prompt_version, cache=cache,
                    extra_tokenizers=extra_tokenizers,
                    paired_b_text=paired_b.text if paired_b else None,
                    paired_b_token_count=paired_b.token_count if paired_b else None,
                )

            if result.valid:
                stats.generated += 1
                if family == "B":
                    b_results[(item_hash, length_bucket, target)] = result
            else:
                stats.failed += 1
                stats.failures.append({
                    "item_hash": item_hash,
                    "family": family,
                    "length_bucket": length_bucket,
                    "target": target,
                    "notes": result.validation_notes,
                })

    return stats
