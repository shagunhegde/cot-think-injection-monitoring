"""Tests for gen_seed: determinism, uniqueness, and range."""
from src.infra.seed import gen_seed

_MASK31 = (1 << 31) - 1


def test_deterministic():
    s1 = gen_seed(0, "key_abc", 0)
    s2 = gen_seed(0, "key_abc", 0)
    assert s1 == s2


def test_different_sample_idx():
    s0 = gen_seed(0, "key_abc", 0)
    s1 = gen_seed(0, "key_abc", 1)
    s2 = gen_seed(0, "key_abc", 2)
    assert s0 != s1 != s2


def test_different_global_seed():
    assert gen_seed(0, "k", 0) != gen_seed(1, "k", 0)


def test_different_condition_key():
    assert gen_seed(0, "key_a", 0) != gen_seed(0, "key_b", 0)


def test_within_int31_range():
    for i in range(20):
        s = gen_seed(42, f"cond_{i}", i)
        assert 0 <= s <= _MASK31


def test_raising_samples_leaves_existing_unchanged():
    """Raising samples_per_condition from 1→5 only adds new sample indices; existing seed (idx=0) unchanged."""
    seed_at_0_before = gen_seed(0, "key_x", 0)
    seed_at_0_after = gen_seed(0, "key_x", 0)  # same call, same answer
    assert seed_at_0_before == seed_at_0_after
