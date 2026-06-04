"""Tests for select_target: most_plausible and sweep modes."""
import math
from src.data.select_target import select_targets, uniform_logprobs

ITEM = {
    "item_hash": "abc",
    "question": "Q",
    "choices": ["opt_a", "opt_b", "opt_c", "opt_d"],
    "answer_idx": 0,      # correct = A
    "answer_letter": "A",
}


def test_most_plausible_picks_highest_wrong():
    # B(idx=1) has highest log-prob among wrong options.
    lp = [0.9, 0.7, 0.3, 0.1]
    targets = select_targets(ITEM, lp, mode="most_plausible")
    assert targets == ["B"]


def test_most_plausible_not_correct():
    # Correct is A (idx=0) with the highest overall prob; should still pick best wrong.
    lp = [10.0, 3.0, 2.0, 1.0]
    targets = select_targets(ITEM, lp, mode="most_plausible")
    assert targets == ["B"]
    assert "A" not in targets


def test_sweep_returns_three_wrong_options():
    lp = [0.9, 0.7, 0.3, 0.1]
    targets = select_targets(ITEM, lp, mode="sweep")
    assert len(targets) == 3
    assert "A" not in targets
    # Ranked descending: B > C > D
    assert targets == ["B", "C", "D"]


def test_sweep_order():
    lp = [0.5, 0.1, 0.8, 0.4]  # correct=A(0.5), wrong: C(0.8) > A skip > B(0.1) < D(0.4)
    targets = select_targets(ITEM, lp, mode="sweep")
    # Wrong options: B(idx=1,0.1), C(idx=2,0.8), D(idx=3,0.4)
    assert targets == ["C", "D", "B"]


def test_uniform_logprobs():
    lp = uniform_logprobs(4)
    assert len(lp) == 4
    assert all(math.isfinite(v) for v in lp)
    # All equal
    assert len(set(lp)) == 1
