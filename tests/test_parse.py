"""Tests for parse.py: trace/answer splitting and letter extraction."""
import pytest
from src.models.parse import split_trace_answer, extract_letter, parse_output


# ── split_trace_answer ────────────────────────────────────────────────────────

def test_split_normal():
    trace, answer = split_trace_answer("<think>\nreasoning here\n</think>\nAnswer: B")
    assert "reasoning" in trace
    assert "Answer: B" in answer

def test_split_no_think_block():
    trace, answer = split_trace_answer("Just answer: A")
    assert trace == ""
    assert "Just answer: A" in answer

def test_split_unclosed_think():
    trace, answer = split_trace_answer("<think>reasoning without close")
    assert "reasoning" in trace
    assert answer == ""

def test_split_strips_think_open_tag():
    trace, _ = split_trace_answer("<think>some trace</think>answer")
    assert not trace.startswith("<think>")


# ── extract_letter ────────────────────────────────────────────────────────────

def test_extract_boxed():
    assert extract_letter(r"The result is \boxed{C}.") == "C"

def test_extract_answer_colon():
    assert extract_letter("Answer: B") == "B"

def test_extract_therefore():
    assert extract_letter("Therefore the answer is D.") == "D"

def test_extract_paren():
    assert extract_letter("I pick (A) as my answer.") == "A"

def test_extract_none():
    assert extract_letter("No letter here at all.") is None

def test_extract_prefer_last():
    text = r"First \boxed{A} then \boxed{C}"
    assert extract_letter(text, prefer_last=True) == "C"
    assert extract_letter(text, prefer_last=False) == "A"

def test_extract_case_insensitive():
    assert extract_letter("answer: b") == "B"


# ── parse_output ──────────────────────────────────────────────────────────────

def test_parse_full():
    output = "<think>\nTherefore the answer is B.\n</think>\n\\boxed{B}"
    r = parse_output(output)
    assert r["boxed_letter"] == "B"
    assert r["trace_letter"] == "B"
    assert r["has_think_block"] is True
    assert r["dissociation"] is False

def test_parse_dissociation():
    """Trace says B, but boxed answer is A — classic trace-answer dissociation."""
    output = "<think>\nTherefore the answer is B.\n</think>\n\\boxed{A}"
    r = parse_output(output)
    assert r["boxed_letter"] == "A"
    assert r["trace_letter"] == "B"
    assert r["dissociation"] is True

def test_parse_no_think_block():
    output = "\\boxed{C}"
    r = parse_output(output)
    assert r["boxed_letter"] == "C"
    assert r["has_think_block"] is False
    assert r["dissociation"] is False
