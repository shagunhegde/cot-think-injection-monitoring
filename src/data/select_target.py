"""Target selection: pick the most-plausible distractor (or sweep all three wrong options).

The #1 confound in prior work: picking target = first option ≠ correct, which often
selects an implausible distractor. We fix this by scoring the model's per-option
likelihood on the clean question and picking the highest-probability wrong option.

Two modes (§5, §4.1):
  most_plausible : pick the single highest-probability wrong option per item.
  sweep          : return all three wrong options (3× cost; use on a subset).

Results go into cache layer 1 keyed by (model_id, item_hash, target_mode).
"""
from __future__ import annotations

from typing import Literal

IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def select_targets(
    item: dict,
    option_logprobs: list[float],
    mode: Literal["most_plausible", "sweep"] = "most_plausible",
) -> list[str]:
    """Given per-option log-probabilities (from the clean forward pass), return
    target letter(s) — the wrong option(s) ranked by descending plausibility.

    `option_logprobs[i]` is the log-probability the model assigns to answer i.
    `item["answer_idx"]` is the correct answer index.
    """
    correct_idx = item["answer_idx"]
    n_options = len(item["choices"])

    # Rank wrong options by descending log-probability (most plausible first).
    wrong_ranked = sorted(
        [i for i in range(n_options) if i != correct_idx],
        key=lambda i: option_logprobs[i],
        reverse=True,
    )

    if mode == "most_plausible":
        return [IDX_TO_LETTER[wrong_ranked[0]]]
    else:  # sweep
        return [IDX_TO_LETTER[i] for i in wrong_ranked]


def score_options_hf(
    item: dict,
    model,
    tokenizer,
    device: str = "cuda",
) -> list[float]:
    """Score each answer option by its log-probability given the question.

    Strategy: for each option letter X, score P(X | question + answer-prefix).
    Returns a list of 4 log-probs, one per option (index matches item["choices"]).

    This runs on the GPU; call from Colab / GPU host only.
    """
    import torch

    question = item["question"]
    choices = item["choices"]

    log_probs: list[float] = []
    for idx, choice in enumerate(choices):
        letter = IDX_TO_LETTER[idx]
        # Prompt format: question + "Answer:" and score the letter token.
        prompt = f"{question}\n\nAnswer: {letter}"
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model(**enc)
        # Log-prob of the answer letter at the last position before the letter token.
        # We use a simpler heuristic: just take the log-prob at the position
        # immediately before the final token (the letter).
        logits = out.logits[0, -2, :]  # shape [vocab]
        log_prob = torch.log_softmax(logits, dim=-1)
        # Token id for the letter.
        letter_id = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if not letter_id:
            letter_id = tokenizer.encode(letter, add_special_tokens=False)
        lp = float(log_prob[letter_id[0]]) if letter_id else 0.0
        log_probs.append(lp)

    return log_probs


def uniform_logprobs(n: int = 4) -> list[float]:
    """Uniform log-probs for use when a model is unavailable (e.g. dry-run / tests)."""
    import math

    return [math.log(1 / n)] * n
