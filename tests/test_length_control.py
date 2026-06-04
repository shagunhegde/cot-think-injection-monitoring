"""Tests for length_control: token counting, range checking, truncation, enforce loop."""
import pytest
import warnings
from src.attacks.length_control import (
    MockTokenizer, token_count, target_range, in_range,
    truncate_to_length, enforce_length,
)

TOK = MockTokenizer()
CONCLUSION = "Therefore the answer is B."


def test_target_range_tolerance():
    lo, hi = target_range(300, 0.12)
    assert lo == 264
    assert hi == 336


def test_token_count_mock():
    # MockTokenizer: 1 token ≈ 0.75 words
    text = " ".join(["word"] * 10)  # 10 words → ~13 tokens
    n = token_count(text, TOK)
    assert n >= 10  # should be >10


def test_in_range_true():
    # Build a text whose mock token count is ~300.
    # MockTokenizer: count ≈ words * 1.33. To get ~300 tokens, need ~225 words.
    text = " ".join(["hello"] * 230) + "\n" + CONCLUSION
    n = token_count(text, TOK)
    lo, hi = target_range(300, 0.12)
    if lo <= n <= hi:
        assert in_range(text, 300, TOK)


def test_in_range_false_short():
    assert not in_range("Short text.", 1000, TOK)


def test_truncate_removes_excess():
    # Build a long text and truncate to a small target.
    long = "The history of tea is fascinating. " * 50 + "\n" + CONCLUSION
    result = truncate_to_length(long, 50, CONCLUSION, TOK)
    n = token_count(result, TOK)
    assert n <= target_range(50, 0.12)[1]
    assert result.strip().endswith(CONCLUSION)


def test_truncate_preserves_conclusion():
    text = "Sentence one. Sentence two. Sentence three.\n" + CONCLUSION
    result = truncate_to_length(text, 20, CONCLUSION, TOK)
    assert result.strip().endswith(CONCLUSION)


def test_truncate_already_short():
    short = "Short.\n" + CONCLUSION
    result = truncate_to_length(short, 1000, CONCLUSION, TOK)
    assert result.strip().endswith(CONCLUSION)


def test_enforce_length_already_in_range():
    # MockTokenizer: round(n_words * 1.33). To land in [88,112] for target=100,
    # need ~80 words → round(80 * 1.33) = round(106.4) = 106 tokens ✓
    text = " ".join(["word"] * 80) + "\n" + CONCLUSION
    result = enforce_length(text, 100, CONCLUSION, TOK, tolerance=0.12)
    lo, hi = target_range(100, 0.12)
    assert result.in_range, f"token_count={result.token_count} not in [{lo},{hi}]"
    assert lo <= result.token_count <= hi
    assert result.failure_reason is None


def test_enforce_length_truncates_long():
    long_text = " ".join(["word"] * 600) + "\n" + CONCLUSION
    result = enforce_length(long_text, 100, CONCLUSION, TOK, tolerance=0.12)
    assert result.text.strip().endswith(CONCLUSION)
    assert result.token_count <= target_range(100, 0.12)[1]


def test_enforce_length_calls_generator_when_short():
    short_text = "Too short.\n" + CONCLUSION
    calls = []

    def gen(target: int) -> str:
        calls.append(target)
        return " ".join(["word"] * 55) + "\n" + CONCLUSION

    result = enforce_length(short_text, 100, CONCLUSION, TOK,
                            generator_fn=gen, max_retries=3)
    assert len(calls) >= 1


def test_enforce_length_no_generator_fails_gracefully():
    short_text = "Too short.\n" + CONCLUSION
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = enforce_length(short_text, 500, CONCLUSION, TOK,
                                generator_fn=None, max_retries=3)
    assert not result.in_range
    assert result.failure_reason is not None
    assert len(w) == 1


def test_enforce_length_records_per_model_counts():
    tok2 = MockTokenizer()
    text = " ".join(["word"] * 55) + "\n" + CONCLUSION
    result = enforce_length(text, 100, CONCLUSION, TOK,
                            extra_tokenizers={"tok2": tok2})
    assert "tok2" in result.per_model_counts
    assert result.per_model_counts["tok2"] > 0
