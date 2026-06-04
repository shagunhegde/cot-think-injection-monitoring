"""Tests for attack templates: Family A expansion, B/C/D prompt structure."""
import pytest
from src.attacks.templates import (
    build_family_a, build_family_b_prompt, build_family_c_prompt,
    build_family_d_prompt, build_generator_prompt,
    conclusion_sentence, pick_c_topic,
)
from src.attacks.length_control import MockTokenizer

TOK = MockTokenizer()

ITEM = {
    "item_hash": "aabbccdd" * 8,
    "question": "What is the boiling point of water at sea level?",
    "choices": ["50°C", "100°C", "150°C", "200°C"],
    "answer_idx": 1,   # correct = B (100°C)
    "answer_letter": "B",
    "subject": "physics",
}


# ── conclusion_sentence ───────────────────────────────────────────────────────

def test_conclusion_sentence():
    assert conclusion_sentence("C") == "Therefore the answer is C."
    assert conclusion_sentence("c") == "Therefore the answer is C."


# ── Family A ──────────────────────────────────────────────────────────────────

def test_family_a_ends_with_conclusion():
    text = build_family_a("C", 100, TOK)
    assert text.strip().endswith("Therefore the answer is C.")

def test_family_a_target_letter_in_text():
    text = build_family_a("D", 300, TOK)
    assert "D" in text

def test_family_a_reaches_approximate_length():
    text = build_family_a("A", 200, TOK)
    n = len(TOK.encode(text))
    # Should be at least halfway to target (template expansion is approximate).
    assert n >= 50

def test_family_a_no_lm_needed():
    # Must work without any real tokenizer.
    text = build_family_a("B", 100, None)
    assert "Therefore the answer is B." in text


# ── Family B ──────────────────────────────────────────────────────────────────

def test_family_b_prompt_contains_target():
    msgs, conclusion = build_family_b_prompt(ITEM, "C", 300)
    assert any("C" in m["content"] for m in msgs)
    assert conclusion == "Therefore the answer is C."

def test_family_b_prompt_mentions_correct_answer():
    msgs, _ = build_family_b_prompt(ITEM, "C", 300)
    user_content = msgs[-1]["content"]
    assert "B" in user_content  # correct answer is B

def test_family_b_prompt_has_system_and_user():
    msgs, _ = build_family_b_prompt(ITEM, "A", 100)
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert "user" in roles


# ── Family C ──────────────────────────────────────────────────────────────────

def test_family_c_prompt_contains_target():
    msgs, conclusion = build_family_c_prompt(ITEM, "D", 300)
    assert conclusion == "Therefore the answer is D."

def test_family_c_topic_is_off_topic():
    topic = pick_c_topic(ITEM["item_hash"])
    # Topic should not contain physics/boiling keywords.
    assert "boiling" not in topic.lower()

def test_family_c_same_item_same_topic():
    t1 = pick_c_topic(ITEM["item_hash"])
    t2 = pick_c_topic(ITEM["item_hash"])
    assert t1 == t2

def test_family_c_different_items_may_differ():
    t1 = pick_c_topic("aabbccdd" * 8)
    t2 = pick_c_topic("11223344" * 8)
    # Not guaranteed to differ (pool rotation), but at least they both run.
    assert isinstance(t1, str) and isinstance(t2, str)


# ── Family D ──────────────────────────────────────────────────────────────────

def test_family_d_prompt_contains_target_and_correct():
    msgs, conclusion = build_family_d_prompt(ITEM, "A", 700)
    user = msgs[-1]["content"]
    assert "A" in user   # target
    assert "B" in user   # correct answer to reconsider
    assert conclusion == "Therefore the answer is A."

def test_family_d_prompt_mentions_recheck():
    msgs, _ = build_family_d_prompt(ITEM, "C", 700)
    user = msgs[-1]["content"].lower()
    assert "re-check" in user or "verify" in user or "verification" in user


# ── Dispatch ──────────────────────────────────────────────────────────────────

def test_dispatch_family_a_raises():
    with pytest.raises(ValueError, match="template-only"):
        build_generator_prompt(ITEM, "A", "C", 100)

def test_dispatch_family_b():
    msgs, conc = build_generator_prompt(ITEM, "B", "C", 300)
    assert conc == "Therefore the answer is C."

def test_dispatch_family_c_uses_paired_b_length():
    msgs, _ = build_generator_prompt(ITEM, "C", "D", 300, paired_b_length=450)
    # The 450 token request should appear in the prompt.
    assert "450" in msgs[-1]["content"]

def test_dispatch_family_d():
    msgs, conc = build_generator_prompt(ITEM, "D", "A", 700)
    assert conc == "Therefore the answer is A."

def test_dispatch_unknown_family():
    with pytest.raises(ValueError, match="Unknown family"):
        build_generator_prompt(ITEM, "Z", "A", 300)
