"""Tests for cost.estimate: zero-cost on empty matrix, non-zero for real matrix, Report.summary."""
from src.infra.cost import estimate, Report, can_proceed
from src.infra.matrix import build_condition_matrix


COST_CFG = {
    "sec_per_gen": {"m1": 10},
    "judge_avg_tokens": 500,
    "judge_price_per_1k": 0.002,
    "prefill_gen_avg_tokens": 800,
    "prefill_gen_price_per_1k": 0.002,
}


def test_empty_matrix_zero_cost():
    report = estimate([], caches={}, cost_cfg=COST_CFG)
    assert report.is_zero()
    assert report.total_usd() == 0.0
    assert report.gpu_hours == 0.0


def test_non_empty_matrix_has_nonzero_cost():
    cfg = {
        "datasets": ["mmlu"],
        "models": ["m1"],
        "families": ["B"],
        "lengths": [100],
        "targets": "most_plausible",
        "monitor_variants": ["full_trace"],
        "monitored_states": ["unmonitored"],
        "n_items": 3,
        "samples_per_condition": 1,
        "seed": 0,
    }
    matrix = build_condition_matrix(cfg)
    report = estimate(matrix, caches={}, cost_cfg=COST_CFG)
    assert report.subject_gens_new == 3
    assert report.gpu_hours > 0
    assert report.api_usd > 0


def test_summary_prints(capsys):
    report = estimate([], caches={}, cost_cfg=COST_CFG)
    report.dry_run = True
    print(report.summary())
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "$" in out


def test_can_proceed_no_cap():
    r = Report(api_usd=1000.0)
    assert can_proceed(r, 0.0, None)


def test_can_proceed_within_cap():
    r = Report(api_usd=5.0)
    assert can_proceed(r, 3.0, 10.0)


def test_can_proceed_exceeds_cap():
    r = Report(api_usd=8.0)
    assert not can_proceed(r, 5.0, 10.0)


def test_family_a_prefills_free(COST_CFG=COST_CFG):
    """Family A uses templates, not API calls — no prefill_gen cost."""
    cfg = {
        "datasets": ["mmlu"], "models": ["m1"], "families": ["A"],
        "lengths": [100], "targets": "most_plausible",
        "monitor_variants": ["full_trace"], "monitored_states": ["unmonitored"],
        "n_items": 2, "samples_per_condition": 1, "seed": 0,
    }
    matrix = build_condition_matrix(cfg)
    # B/C/D charge prefill API; A should not add prefill_gen cost.
    report_a = estimate(matrix, caches={}, cost_cfg=COST_CFG)

    cfg_b = {**cfg, "families": ["B"]}
    matrix_b = build_condition_matrix(cfg_b)
    report_b = estimate(matrix_b, caches={}, cost_cfg=COST_CFG)

    assert report_b.api_usd > report_a.api_usd  # B charges for prefill gen
