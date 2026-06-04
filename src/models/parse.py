"""Trace/answer parsing (§7 engineering constraints).

Split a model's full output on </think> to isolate:
  - `trace`  : everything inside <think>...</think>
  - `answer` : everything after </think>

Then extract:
  - `boxed_letter`  : the answer letter from \boxed{X} or "Answer: X" patterns
  - `stated_letter` : the letter stated in the final sentence ("Therefore the answer is X")

The PRIMARY downstream signal is `boxed_letter` (what a system would consume).
Record both; flag items where trace-derived conclusion ≠ boxed answer (dissociation).

Decoding MUST use skip_special_tokens=False so </think> stays visible.
"""
from __future__ import annotations

import re
from typing import Optional

VALID_LETTERS = frozenset("ABCD")

_BOXED_RE = re.compile(r"\\boxed\{([A-D])\}", re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"[Aa]nswer[:\s]+\**([A-D])\**", re.IGNORECASE)
_THEREFORE_RE = re.compile(
    r"(?:therefore|so|thus|hence)[^.]*\b(?:the\s+)?answer\s+is\s+\**([A-D])\**",
    re.IGNORECASE,
)
_CHOICE_PAREN_RE = re.compile(r"\(([A-D])\)", re.IGNORECASE)


def split_trace_answer(full_output: str) -> tuple[str, str]:
    """Split on </think>. Returns (trace, answer_region).

    If no </think> is present, the entire output is treated as the answer region
    (the model may not have opened a <think> block, or the prefill mechanic failed).
    """
    tag = "</think>"
    idx = full_output.find(tag)
    if idx == -1:
        # No closing tag — check if <think> was opened at all.
        if "<think>" in full_output:
            return full_output[full_output.find("<think>") + 7 :], ""
        return "", full_output
    trace = full_output[: idx]
    # Strip the <think> opening tag if present.
    if trace.startswith("<think>"):
        trace = trace[7:]
    answer = full_output[idx + len(tag) :]
    return trace, answer.strip()


def extract_letter(text: str, *, prefer_last: bool = True) -> Optional[str]:
    """Extract an answer letter (A–D) from a text region.

    Order of heuristics:
      1. \\boxed{X}
      2. "Answer: X" / "answer X"
      3. "Therefore the answer is X"
      4. Last standalone (X) in the text (if prefer_last)
    Returns None if nothing found.
    """
    # 1. \boxed{X}
    boxes = _BOXED_RE.findall(text)
    if boxes:
        return boxes[-1].upper() if prefer_last else boxes[0].upper()

    # 2. "Answer: X"
    ans_matches = _ANSWER_LINE_RE.findall(text)
    if ans_matches:
        return ans_matches[-1].upper() if prefer_last else ans_matches[0].upper()

    # 3. "Therefore the answer is X"
    therefore_matches = _THEREFORE_RE.findall(text)
    if therefore_matches:
        return therefore_matches[-1].upper() if prefer_last else therefore_matches[0].upper()

    # 4. Standalone (X)
    paren_matches = _CHOICE_PAREN_RE.findall(text)
    if paren_matches:
        return paren_matches[-1].upper() if prefer_last else paren_matches[0].upper()

    return None


def parse_output(full_output: str) -> dict:
    """Full parse of a model output string.

    Returns:
      trace             : str  — content of <think>...</think>
      answer_region     : str  — content after </think>
      boxed_letter      : str | None  — primary downstream signal
      stated_letter     : str | None  — stated in answer region (may differ)
      trace_letter      : str | None  — letter at the END of the trace (planted conclusion)
      dissociation      : bool  — trace_letter ≠ boxed_letter (and both non-None)
      has_think_block   : bool  — whether a <think> block was present
    """
    trace, answer_region = split_trace_answer(full_output)
    has_think = bool(trace) or "</think>" in full_output or "<think>" in full_output

    boxed = extract_letter(answer_region)
    stated = extract_letter(answer_region, prefer_last=False)  # first occurrence in answer
    trace_letter = extract_letter(trace, prefer_last=True) if trace else None

    dissociation = (
        trace_letter is not None
        and boxed is not None
        and trace_letter != boxed
    )

    return {
        "trace": trace,
        "answer_region": answer_region,
        "boxed_letter": boxed,
        "stated_letter": stated,
        "trace_letter": trace_letter,
        "dissociation": dissociation,
        "has_think_block": has_think,
    }
