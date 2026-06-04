"""Tests for build_condition_matrix: all filters, samples_per_condition, family-length validity."""
import pytest

from src.infra.matrix import build_condition_matrix, valid_family_length


# ── valid_family_length ──────────────────────────────────────────────────────

def test_family_a_max_700():
    assert valid_family_length("A", 700)
    assert not valid_family_length("A", 1500)

def test_family_d_min_300():
    assert valid_family_length("D", 300)
    assert not valid_family_length("D", 100)

def test_family_b_full_range():
    for l in [100, 300, 700, 1500]:
        assert valid_family_length("B", l)

def test_length_zero_always_false():
    for f in ("A", "B", "C", "D"):
        assert not valid_family_length(f, 0)


# ── build_condition_matrix ────────────────────────────────────────────────────

BASE_CFG = {
    "datasets": ["mmlu"],
    "models": ["m1"],
    "families": ["B"],
    "lengths": [100, 300],
    "targets": "most_plausible",
    "monitor_variants": ["full_trace"],
    "monitored_states": ["unmonitored"],
    "n_items": 2,
    "samples_per_condition": 1,
    "seed": 0,
}


def test_basic_matrix_size():
    matrix = build_condition_matrix(BASE_CFG)
    # 1 dataset × 2 items × 1 model × 1 family × 2 lengths × 1 target × 1 variant × 1 state × 1 sample
    assert len(matrix) == 4


def test_samples_per_condition():
    cfg = {**BASE_CFG, "samples_per_condition": 3}
    matrix = build_condition_matrix(cfg)
    assert len(matrix) == 4 * 3  # 3 samples per condition

    sample_indices = {c["sample_idx"] for c in matrix}
    assert sample_indices == {0, 1, 2}


def test_family_d_length_100_excluded():
    cfg = {**BASE_CFG, "families": ["D"], "lengths": [100, 300, 700]}
    matrix = build_condition_matrix(cfg)
    lengths_in = {c["length"] for c in matrix}
    assert 100 not in lengths_in     # D min is 300
    assert 300 in lengths_in
    assert 700 in lengths_in


def test_family_a_length_1500_excluded():
    cfg = {**BASE_CFG, "families": ["A"], "lengths": [700, 1500]}
    matrix = build_condition_matrix(cfg)
    lengths_in = {c["length"] for c in matrix}
    assert 1500 not in lengths_in    # A max is 700
    assert 700 in lengths_in


def test_limit_cap():
    cfg = {**BASE_CFG, "n_items": 10, "lengths": [100, 300, 700], "limit": 3}
    matrix = build_condition_matrix(cfg)
    assert len(matrix) == 3


def test_max_conditions_cap():
    cfg = {**BASE_CFG, "n_items": 10, "lengths": [100, 300, 700], "max_conditions": 5}
    matrix = build_condition_matrix(cfg)
    assert len(matrix) == 5


def test_sample_fraction():
    cfg = {**BASE_CFG, "n_items": 20, "sample_fraction": 0.5, "seed": 42}
    full = build_condition_matrix({**BASE_CFG, "n_items": 20})
    sampled = build_condition_matrix(cfg)
    assert len(sampled) < len(full)
    assert len(sampled) == max(1, round(len(full) * 0.5))


def test_sample_fraction_reproducible():
    cfg = {**BASE_CFG, "n_items": 20, "sample_fraction": 0.5, "seed": 7}
    m1 = build_condition_matrix(cfg)
    m2 = build_condition_matrix(cfg)
    assert [c["condition_key"] for c in m1] == [c["condition_key"] for c in m2]


def test_condition_key_present_and_stable():
    matrix = build_condition_matrix(BASE_CFG)
    for cond in matrix:
        assert "condition_key" in cond
        assert len(cond["condition_key"]) == 64

    # Same conditions → same keys.
    matrix2 = build_condition_matrix(BASE_CFG)
    keys1 = {c["condition_key"] for c in matrix}
    keys2 = {c["condition_key"] for c in matrix2}
    assert keys1 == keys2


def test_empty_models_returns_empty():
    cfg = {**BASE_CFG, "models": []}
    assert build_condition_matrix(cfg) == []


def test_n_items_zero_returns_empty():
    cfg = {**BASE_CFG, "n_items": 0}
    assert build_condition_matrix(cfg) == []


def test_multi_model_multi_variant():
    cfg = {
        **BASE_CFG,
        "models": ["m1", "m2"],
        "monitor_variants": ["full_trace", "answer_only"],
        "n_items": 1,
        "lengths": [100],
    }
    matrix = build_condition_matrix(cfg)
    # 1 item × 2 models × 1 family × 1 length × 1 target × 2 variants × 1 state × 1 sample
    assert len(matrix) == 4
