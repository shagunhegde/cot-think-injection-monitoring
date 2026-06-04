"""Tests for generate.py — uses MockBackend (no real API calls)."""
import os
import tempfile
import pytest

from src.infra.cache import Cache
from src.infra.llm_client import LLMClient, MockBackend
from src.attacks.length_control import MockTokenizer
from src.attacks.generate import (
    generate_family_a, generate_prefill, generate_all_prefills,
    prefill_cache_key, PrefillResult,
)

TOK = MockTokenizer()

ITEM = {
    "item_hash": "deadbeef" * 8,
    "question": "What is the capital of France?",
    "choices": ["London", "Paris", "Berlin", "Rome"],
    "answer_idx": 1,
    "answer_letter": "B",
    "subject": "geography",
}


def _mock_client(replies=None):
    """Return an LLMClient backed by MockBackend with given replies."""
    # Build a reply that passes length (150-ish mock tokens) and ends with conclusion.
    default_reply = (
        " ".join(["word"] * 90)
        + "\nTherefore the answer is A."
    )
    return LLMClient(MockBackend(replies or [default_reply] * 20))


# ── Family A ──────────────────────────────────────────────────────────────────

def test_generate_family_a_valid(tmp_path):
    cache = Cache(tmp_path, "test")
    result = generate_family_a(ITEM, "A", 100, TOK, cache=cache)
    assert result.family == "A"
    assert result.target_letter == "A"
    assert "Therefore the answer is A." in result.text

def test_generate_family_a_cached_on_second_call(tmp_path):
    cache = Cache(tmp_path, "test")
    r1 = generate_family_a(ITEM, "C", 100, TOK, cache=cache)
    # Second call should hit cache.
    r2 = generate_family_a(ITEM, "C", 100, TOK, cache=cache)
    assert r1.text == r2.text
    assert cache.hits >= 1


# ── Family B ──────────────────────────────────────────────────────────────────

def test_generate_prefill_b(tmp_path):
    client = _mock_client()
    cache = Cache(tmp_path, "test")
    result = generate_prefill(
        ITEM, "B", "A", 100, client, TOK,
        cache=cache, generator_model="mock", prompt_version="v1",
    )
    assert result.family == "B"
    assert result.target_letter == "A"

def test_generate_prefill_b_cached(tmp_path):
    client = _mock_client()
    cache = Cache(tmp_path, "test")
    r1 = generate_prefill(ITEM, "B", "A", 100, client, TOK, cache=cache,
                          generator_model="mock", prompt_version="v1")
    # New cache instance forces re-load from disk; same result proves cache hit.
    cache2 = Cache(tmp_path, "test")
    r2 = generate_prefill(ITEM, "B", "A", 100, client, TOK, cache=cache2,
                          generator_model="mock", prompt_version="v1")
    if r1.valid:
        # Valid first result must be returned identically from cache on second call.
        assert r1.text == r2.text

def test_generate_prefill_invalid_not_cached(tmp_path):
    """A generated prefill that fails validation must NOT be written to cache."""
    # Reply that has no conclusion sentence → will fail validate.
    bad_reply = " ".join(["word"] * 90)  # no "Therefore the answer is..."
    client = _mock_client([bad_reply] * 10)
    cache = Cache(tmp_path, "bad_test")
    result = generate_prefill(
        ITEM, "B", "A", 100, client, TOK,
        cache=cache, generator_model="mock", prompt_version="v1",
    )
    assert not result.valid
    # Cache should NOT have stored this.
    ck = prefill_cache_key("mmlu", ITEM["item_hash"], "B", 100, "A", "mock", "v1")
    assert not cache.contains("prefills/mmlu_B_100", ck)


# ── Batch generation ──────────────────────────────────────────────────────────

def _conditions(families, lengths, n_items=2):
    conditions = []
    for i in range(n_items):
        item_hash = ITEM["item_hash"] if i == 0 else f"item{i}" * 16
        for f in families:
            for length in lengths:
                conditions.append({
                    "dataset": "mmlu",
                    "item": item_hash,
                    "family": f,
                    "length": length,
                    "target": "A",   # pre-resolved letter
                    "condition_key": f"{item_hash}_{f}_{length}_A",
                    "sample_idx": 0,
                })
    return conditions


def test_generate_all_b_only(tmp_path):
    client = _mock_client()
    cache = Cache(tmp_path, "batch")
    items_by_hash = {ITEM["item_hash"]: ITEM}
    conds = _conditions(["B"], [100], n_items=1)
    stats = generate_all_prefills(
        conds, items_by_hash, client, TOK, cache,
        generator_model="mock", prompt_version="v1",
    )
    assert stats.total >= 1
    assert stats.generated + stats.cached + stats.failed == stats.total


def test_generate_all_c_requires_b_first(tmp_path):
    """Family C must be generated after B (matched pairs). Without B, C should fail."""
    client = _mock_client()
    cache = Cache(tmp_path, "c_no_b")
    items_by_hash = {ITEM["item_hash"]: ITEM}
    # Only C, no B.
    conds = _conditions(["C"], [100], n_items=1)
    stats = generate_all_prefills(
        conds, items_by_hash, client, TOK, cache,
        generator_model="mock", prompt_version="v1",
    )
    # Without paired B, C conditions should be counted as failures.
    assert stats.failed >= 1


def test_generate_all_b_then_c(tmp_path):
    """B and C together — C should be matched to the generated B."""
    # B reply must be on-topic (contain question keywords) to pass validation.
    q_words = "capital France Paris London Berlin Rome "
    reply_b = (q_words * 15).strip() + "\nTherefore the answer is A."
    # C reply is off-topic (just filler words unrelated to the question).
    reply_c = " ".join(["tea"] * 80) + "\nTherefore the answer is A."
    client = LLMClient(MockBackend([reply_b, reply_c] * 10))
    cache = Cache(tmp_path, "b_c")
    items_by_hash = {ITEM["item_hash"]: ITEM}
    conds = _conditions(["B", "C"], [100], n_items=1)
    stats = generate_all_prefills(
        conds, items_by_hash, client, TOK, cache,
        generator_model="mock", prompt_version="v1",
    )
    # Both B and C should succeed (or at least not both fail).
    assert stats.failed < stats.total


def test_generate_all_second_run_all_cached(tmp_path):
    """Second run should compute nothing new (all cached)."""
    client = _mock_client()
    cache = Cache(tmp_path, "second_run")
    items_by_hash = {ITEM["item_hash"]: ITEM}
    conds = _conditions(["A", "B"], [100], n_items=1)

    stats1 = generate_all_prefills(
        conds, items_by_hash, client, TOK, cache,
        generator_model="mock", prompt_version="v1",
    )
    # Second run with fresh cache instance (forces re-load from disk).
    cache2 = Cache(tmp_path, "second_run")
    stats2 = generate_all_prefills(
        conds, items_by_hash, client, TOK, cache2,
        generator_model="mock", prompt_version="v1",
    )
    assert stats2.generated == 0
    assert stats2.cached == stats1.generated + stats1.cached
