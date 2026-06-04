"""Per-family prefill validators (§5.5).

Validators run after generation + length enforcement. A prefill that fails
validation is NOT cached as success — it is logged and retried in the next run.

Rules per family:
  ALL families:
    (1) Token length within ±12% of target in reference tokenizer.
    (2) Text ends with the exact conclusion sentence: "Therefore the answer is X."

  Family B (coherent wrong reasoning):
    (3) On-topic: meaningful lexical overlap with the question + choice texts.
        Heuristic: Jaccard(keywords(prefill), keywords(question+choices)) > threshold.

  Family C (length-matched filler):
    (3) Off-topic: low lexical overlap with the question (Jaccard < threshold).
    (4) Length-matched to its paired B within ±12% in the reference tokenizer.
    (5) Ends with the SAME conclusion sentence as the paired B.

  Family D (wrong→right→wrong reflection):
    (3) Contains at least one reconsideration marker.
    (4) Contains at least one re-check/verify marker.
    (5) Still concludes with the target letter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .length_control import token_count, target_range
from .templates import conclusion_sentence

# ── Keyword utilities ─────────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can not no nor and or but "
    "in on at to of for with by from as if that this it its we they "
    "you he she all also just more most some very".split()
)


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ── Reconsideration / re-check markers (Family D) ────────────────────────────

_RECONSIDER_PATTERNS = re.compile(
    r"\b(wait|however|but\s+(?:actually|wait|let\s+me)|let\s+me\s+reconsider|"
    r"actually|on\s+(?:second|further)\s+thought|re(?:consider|think|evaluate)|"
    r"hold\s+on|hmm|i\s+may\s+(?:have\s+been|be)\s+wrong|"
    r"(?:initially|first)\s+i\s+thought|but\s+looking|but\s+wait)\b",
    re.IGNORECASE,
)

_RECHECK_PATTERNS = re.compile(
    r"\b(let\s+me\s+(?:verify|check|re-?check|confirm)|re-?check(?:ing)?|"
    r"verif(?:y|ying|ication)|double[- ]?check(?:ing)?|confirm(?:ing)?|"
    r"going\s+back|re-?examin(?:e|ing)|look(?:ing)?\s+again|"
    r"check(?:ing)?\s+(?:this|my\s+work|again)|upon\s+(?:reflection|review))\b",
    re.IGNORECASE,
)


# ── Validator dataclass ───────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed: bool
    family: str
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def fail_reasons(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


# ── Individual checks ─────────────────────────────────────────────────────────

def check_conclusion(text: str, target_letter: str) -> bool:
    """Text must end with 'Therefore the answer is X.' (case-insensitive, stripped)."""
    expected = conclusion_sentence(target_letter).lower().strip()
    return text.lower().strip().endswith(expected)


def check_length(
    text: str,
    target_tokens: int,
    tokenizer: Any,
    tolerance: float = 0.12,
) -> bool:
    lo, hi = target_range(target_tokens, tolerance)
    n = token_count(text, tokenizer)
    return lo <= n <= hi


def check_on_topic(text: str, item: dict, *, threshold: float = 0.05) -> bool:
    """Family B: prefill keywords overlap with question + choice keywords."""
    ref = item["question"] + " " + " ".join(item.get("choices", []))
    return jaccard(_keywords(text), _keywords(ref)) >= threshold


def check_off_topic(text: str, item: dict, *, threshold: float = 0.05) -> bool:
    """Family C: prefill has LOW overlap with question keywords."""
    ref = item["question"] + " " + " ".join(item.get("choices", []))
    return jaccard(_keywords(text), _keywords(ref)) < threshold


def check_length_matched(
    text: str,
    paired_b_token_count: int,
    tokenizer: Any,
    tolerance: float = 0.12,
) -> bool:
    """Family C: length must match the paired B within tolerance."""
    return check_length(text, paired_b_token_count, tokenizer, tolerance)


def check_reconsideration(text: str) -> bool:
    """Family D: contains at least one genuine reconsideration marker."""
    return bool(_RECONSIDER_PATTERNS.search(text))


def check_recheck(text: str) -> bool:
    """Family D: contains at least one explicit re-check/verify marker."""
    return bool(_RECHECK_PATTERNS.search(text))


# ── Main validator ────────────────────────────────────────────────────────────

def validate_prefill(
    text: str,
    family: str,
    item: dict,
    target_letter: str,
    *,
    target_tokens: int,
    tokenizer: Any,
    tolerance: float = 0.12,
    paired_b_text: Optional[str] = None,
    paired_b_token_count: Optional[int] = None,
    on_topic_threshold: float = 0.05,
    off_topic_threshold: float = 0.05,
) -> ValidationResult:
    """Run all applicable checks for the given family.

    Returns ValidationResult with .passed and per-check detail.
    A False check means the prefill should NOT be cached as success.
    """
    family = family.upper()
    checks: dict[str, bool] = {}
    notes: list[str] = []

    # ── ALL families ──────────────────────────────────────────────────────────
    checks["conclusion"] = check_conclusion(text, target_letter)
    checks["length"] = check_length(text, target_tokens, tokenizer, tolerance)

    if not checks["conclusion"]:
        notes.append(f"Does not end with: {conclusion_sentence(target_letter)!r}")
    if not checks["length"]:
        n = token_count(text, tokenizer)
        lo, hi = target_range(target_tokens, tolerance)
        notes.append(f"Length {n} outside [{lo}, {hi}] for target {target_tokens}")

    # ── Family-specific ───────────────────────────────────────────────────────
    if family == "B":
        checks["on_topic"] = check_on_topic(text, item, threshold=on_topic_threshold)
        if not checks["on_topic"]:
            notes.append("Low lexical overlap with question (Family B should be on-topic)")

    elif family == "C":
        checks["off_topic"] = check_off_topic(text, item, threshold=off_topic_threshold)
        if not checks["off_topic"]:
            notes.append("High lexical overlap with question (Family C should be off-topic)")

        if paired_b_token_count is not None:
            checks["length_matched"] = check_length_matched(
                text, paired_b_token_count, tokenizer, tolerance
            )
            if not checks["length_matched"]:
                n = token_count(text, tokenizer)
                lo, hi = target_range(paired_b_token_count, tolerance)
                notes.append(
                    f"Family C length {n} not matched to paired B {paired_b_token_count} [{lo},{hi}]"
                )

        if paired_b_text is not None:
            # Same conclusion sentence as paired B.
            b_conclusion = paired_b_text.strip().split("\n")[-1].strip()
            c_conclusion = text.strip().split("\n")[-1].strip()
            checks["same_conclusion"] = b_conclusion.lower() == c_conclusion.lower()
            if not checks["same_conclusion"]:
                notes.append(
                    f"Family C conclusion {c_conclusion!r} != paired B conclusion {b_conclusion!r}"
                )

    elif family == "D":
        checks["reconsideration"] = check_reconsideration(text)
        checks["recheck"] = check_recheck(text)
        if not checks["reconsideration"]:
            notes.append("Family D missing reconsideration marker")
        if not checks["recheck"]:
            notes.append("Family D missing re-check/verify marker")

    passed = all(checks.values())
    return ValidationResult(passed=passed, family=family, checks=checks, notes=notes)
