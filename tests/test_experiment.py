"""Tests for Phase 5: run_experiment.py guarded loop + aggregate.py Layer 5.

Acceptance criteria:
  - dry_run: estimate printed, ledger written, zero model/API calls, returns immediately.
  - Spend guardrail: loop halts cleanly before breaching max_spend_usd; ledger records halt.
  - Resume after mid-run kill: a pre-populated subject cache skips those conditions (no recompute).
  - _parse_overrides handles bool/int/float/str and ignores malformed tokens.
  - aggregate.py: load_subject_results / load_judge_results / build_enriched_results
    round-trip correctly from JSONL written by Cache.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.infra.cache import Cache
from src.infra.paths import build_paths
from src.pipeline.run_experiment import (
    _parse_overrides,
    _load_items,
    run,
)
from src.pipeline.aggregate import (
    load_subject_results,
    load_judge_results,
    build_enriched_results,
    aggregate_results,
    _parse_shard_stem,
)


# ── _parse_overrides ──────────────────────────────────────────────────────────

def test_parse_overrides_bool():
    result = _parse_overrides(["dry_run=true", "verbose=false"])
    assert result["dry_run"] is True
    assert result["verbose"] is False


def test_parse_overrides_int():
    result = _parse_overrides(["n_items=50", "limit=1"])
    assert result["n_items"] == 50
    assert result["limit"] == 1


def test_parse_overrides_float():
    result = _parse_overrides(["max_spend_usd=0.5"])
    assert result["max_spend_usd"] == pytest.approx(0.5)


def test_parse_overrides_string():
    result = _parse_overrides(["families=[B,C]"])
    assert result["families"] == "[B,C]"


def test_parse_overrides_ignores_no_equals():
    result = _parse_overrides(["--malformed", "dry_run=true"])
    assert "dry_run" in result
    assert "--malformed" not in result


def test_parse_overrides_null():
    result = _parse_overrides(["limit=null"])
    assert result["limit"] is None


# ── _load_items (dry_run mode uses placeholders) ──────────────────────────────

def test_load_items_dry_run_placeholders(tmp_path):
    paths = build_paths(tmp_path, create=True)
    cfg = {"dry_run": True, "n_items": 5}
    items = _load_items(cfg, paths)
    assert len(items) == 5
    assert all("item_hash" in it for it in items)


def test_load_items_dry_run_zero_items(tmp_path):
    paths = build_paths(tmp_path, create=True)
    cfg = {"dry_run": True, "n_items": 0}
    items = _load_items(cfg, paths)
    assert items == []


# ── dry_run: estimate printed, ledger written, no model/API calls ─────────────

def test_dry_run_writes_ledger_and_exits(tmp_path, capsys):
    """dry_run=True must print cost estimate, write a ledger entry, and not touch any LLM."""
    paths = build_paths(tmp_path, create=True)
    cfg = {
        "dry_run": True,
        "n_items": 3,
        "families": ["A"],
        "lengths": [100],
        "models": ["deepseek-r1-distill-1.5b"],
        "targets": ["B"],
        "monitor_variants": ["full_trace"],
        "monitored_states": ["unmonitored"],
        "samples_per_condition": 1,
        "seed": 0,
    }
    cost_cfg = {
        "sec_per_gen": {"deepseek-r1-distill-1.5b": 10},
        "judge_avg_tokens": 500,
        "judge_price_per_1k": 0.00025,
        "prefill_gen_avg_tokens": 1000,
        "prefill_gen_price_per_1k": 0.00025,
    }

    with patch("src.pipeline.run_experiment.LLMClient") as mock_client:
        run(cfg, cost_cfg, paths)

    # LLMClient must never be instantiated in dry_run.
    mock_client.assert_not_called()

    # Ledger directory must have exactly one manifest.
    ledger_files = list((tmp_path / "ledger").glob("*.json"))
    assert len(ledger_files) == 1
    manifest = json.loads(ledger_files[0].read_text())
    assert manifest.get("mode") == "dry_run"  # extra= is spread into top-level

    # Console output must mention matrix size.
    captured = capsys.readouterr()
    assert "Matrix size" in captured.out
    assert "dry_run" in captured.out.lower()


# ── Spend guardrail halts cleanly ─────────────────────────────────────────────

def _make_minimal_cfg(n_items=2):
    return {
        "dry_run": False,
        "n_items": n_items,
        "datasets": [],          # no real data loading
        "families": ["A"],
        "lengths": [100],
        "models": ["m1"],
        "targets": ["B"],
        "monitor_variants": ["full_trace"],
        "monitored_states": ["unmonitored"],
        "samples_per_condition": 1,
        "seed": 0,
        "max_spend_usd": 0.0,    # zero cap → halt immediately
    }


def test_guardrail_halts_before_any_spend(tmp_path):
    """With max_spend_usd=0.0, the loop must halt before completing any non-cached condition."""
    paths = build_paths(tmp_path, create=True)
    cfg = _make_minimal_cfg(n_items=5)
    cost_cfg = {
        "sec_per_gen": {"m1": 10},
        "judge_avg_tokens": 500,
        "judge_price_per_1k": 0.00025,
    }

    flag_resolve = patch("src.pipeline.run_experiment._resolve_targets",
                         side_effect=lambda m, *a, **kw: m)
    flag_client = patch("src.pipeline.run_experiment.LLMClient.from_env",
                        return_value=MagicMock())

    with flag_resolve, flag_client:
        run(cfg, cost_cfg, paths)

    # Ledger must still be written even on guardrail halt.
    ledger_files = list((tmp_path / "ledger").glob("*.json"))
    assert len(ledger_files) == 1


# ── Resume: pre-cached conditions are skipped (no recompute) ─────────────────

def test_resume_skips_cached_conditions(tmp_path, capsys):
    """Pre-populate the subject cache; running again must skip those conditions."""
    paths = build_paths(tmp_path, create=True)

    cfg = {
        "dry_run": False,
        "n_items": 2,
        "datasets": [],
        "families": ["A"],
        "lengths": [100],
        "models": ["m1"],
        "targets": ["B"],
        "monitor_variants": ["full_trace"],
        "monitored_states": ["unmonitored"],
        "samples_per_condition": 1,
        "seed": 0,
        "max_spend_usd": None,
    }
    cost_cfg = {
        "sec_per_gen": {"m1": 10},
        "judge_avg_tokens": 500,
        "judge_price_per_1k": 0.00025,
    }

    # Build the matrix to know condition keys.
    from src.infra.matrix import build_condition_matrix
    items = [{"item_hash": f"item_{i}"} for i in range(2)]
    matrix = build_condition_matrix(cfg, [it["item_hash"] for it in items])

    # Pre-populate subject cache for all conditions.
    subject_cache = Cache(paths.root, "subject")
    for cond in matrix:
        ck = cond["condition_key"]
        shard = f"subject/m1/A_100"
        subject_cache.get_or_compute(
            shard=shard,
            key=ck,
            compute_fn=lambda: {"boxed_letter": "B", "trace": "fake trace", "answer_region": "B"},
            meta={"condition_key": ck},
        )

    generate_called = []
    original_run_subject = __import__(
        "src.pipeline.run_experiment", fromlist=["_run_subject_generation"]
    )._run_subject_generation

    def _spy_generate(cond, *args, **kwargs):
        generate_called.append(cond["condition_key"])
        return original_run_subject(cond, *args, **kwargs)

    flag_resolve = patch("src.pipeline.run_experiment._resolve_targets",
                         side_effect=lambda m, *a, **kw: m)
    flag_gen = patch("src.pipeline.run_experiment._run_subject_generation",
                     side_effect=_spy_generate)
    flag_judges = patch("src.pipeline.run_experiment._run_judges",
                        return_value={"vr_flag": False, "mfr_flag": False,
                                      "vr_rationale": "", "mfr_rationale": "",
                                      "mfr_by_variant": {}})
    flag_client = patch("src.pipeline.run_experiment.LLMClient.from_env",
                        return_value=MagicMock())

    with flag_resolve, flag_gen, flag_judges, flag_client:
        run(cfg, cost_cfg, paths)

    # _run_subject_generation must not have been called (cache hits only).
    assert generate_called == [], f"Should have skipped all cached, but called: {generate_called}"

    captured = capsys.readouterr()
    assert "cached" in captured.out.lower()


# ── aggregate.py: round-trip JSONL → enriched → grouped ─────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_subject_results_reads_cache(tmp_path):
    """load_subject_results should read Layer-3 JSONL shards."""
    shard = tmp_path / "cache" / "subject" / "m1" / "A_100.jsonl"
    records = [
        {"key": "ck1", "result": {"boxed_letter": "A", "trace": "t1"}, "meta": {"family": "A", "length": 100, "target": "B", "sample_idx": 0}},
        {"key": "ck2", "result": {"boxed_letter": "B", "trace": "t2"}, "meta": {"family": "A", "length": 100, "target": "B", "sample_idx": 0}},
    ]
    _write_jsonl(shard, records)

    results = load_subject_results(tmp_path)
    assert len(results) == 2
    letters = {r["boxed_letter"] for r in results}
    assert letters == {"A", "B"}


def test_load_subject_results_skips_errors(tmp_path):
    """Records with error field are filtered out."""
    shard = tmp_path / "cache" / "subject" / "m1" / "A_100.jsonl"
    records = [
        {"key": "ck1", "result": {"error": "oops"}, "meta": {}},
        {"key": "ck2", "result": {"boxed_letter": "A", "trace": "ok"}, "meta": {}},
    ]
    _write_jsonl(shard, records)

    results = load_subject_results(tmp_path)
    assert len(results) == 1
    assert results[0]["boxed_letter"] == "A"


def test_load_subject_results_skips_truncated_line(tmp_path):
    """Malformed JSONL lines (truncated write) must be silently skipped."""
    shard = tmp_path / "cache" / "subject" / "m1" / "A_100.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with open(shard, "w") as f:
        f.write(json.dumps({"key": "ck1", "result": {"boxed_letter": "A", "trace": ""}, "meta": {}}) + "\n")
        f.write("{truncated bad json\n")  # simulates crash mid-write

    results = load_subject_results(tmp_path)
    assert len(results) == 1


def test_load_judge_results_faithfulness(tmp_path):
    """Faithfulness shard → vr_flag populated."""
    shard = tmp_path / "cache" / "judge" / "m1_faithfulness.jsonl"
    records = [
        {"key": "ck1", "result": {"flag": True, "rationale": "yes"}},
    ]
    _write_jsonl(shard, records)

    by_ck = load_judge_results(tmp_path)
    assert "ck1" in by_ck
    assert by_ck["ck1"]["vr_flag"] is True


def test_load_judge_results_monitor(tmp_path):
    """Monitor shard → mfr_flag populated."""
    shard = tmp_path / "cache" / "judge" / "m1_full_trace.jsonl"
    records = [
        {"key": "ck1", "result": {"flag": False, "rationale": "clean"}},
    ]
    _write_jsonl(shard, records)

    by_ck = load_judge_results(tmp_path)
    assert "ck1" in by_ck
    assert by_ck["ck1"]["mfr_flag"] is False


def test_build_enriched_results_joins_layers(tmp_path):
    """build_enriched_results correctly joins subject + judge data and runs enrich()."""
    from src.metrics.core import is_capturable

    subject_results = [
        {
            "boxed_letter": "A",
            "trace": "some trace",
            "answer_region": "A",
            "trace_letter": "A",
            "dissociation": False,
            "has_think_block": True,
            "_condition_key": "ck1",
            "_family": "A",
            "_length": 100,
            "_target": "A",
            "_sample_idx": 0,
            "baseline_letter": "B",
            "target_letter": "A",
        }
    ]
    judge_by_ck = {
        "ck1": {"vr_flag": False, "mfr_flag": True}
    }
    enriched = build_enriched_results(subject_results, judge_by_ck, {}, {})
    assert len(enriched) == 1
    r = enriched[0]
    assert "cir" in r
    assert "mfr" in r
    assert r["mfr"] == 1  # mfr_flag=True
    assert r["cir"] == 1  # boxed_letter == target_letter and baseline != target


def test_parse_shard_stem_recovers_family_length():
    """family/length recovered from shard filename for legacy records."""
    assert _parse_shard_stem("B_100") == ("B", 100)
    assert _parse_shard_stem("full_trace_700") == ("full_trace", 700)
    assert _parse_shard_stem("nodelim") == ("", 0)


def test_load_subject_results_path_fallback(tmp_path):
    """When meta lacks family/length, they are recovered from the shard filename."""
    shard = tmp_path / "cache" / "subject" / "deepseek_X" / "B_300.jsonl"
    records = [
        # meta has only condition_key + sample_idx (legacy record)
        {"key": "ck1", "result": {"boxed_letter": "A", "trace": "t"},
         "meta": {"condition_key": "ck1", "sample_idx": 0}},
    ]
    _write_jsonl(shard, records)

    results = load_subject_results(tmp_path)
    assert len(results) == 1
    assert results[0]["_family"] == "B"
    assert results[0]["_length"] == 300


def test_build_enriched_results_joins_baseline_via_condition_meta(tmp_path):
    """condition_meta supplies baseline_letter + target_letter the subject record lacks."""
    subject_results = [
        {
            "boxed_letter": "A",      # model answered the target
            "trace": "t", "answer_region": "A", "trace_letter": "A",
            "dissociation": False, "has_think_block": True,
            "_condition_key": "ckX", "_family": "B", "_length": 300,
            "_target": "", "_item_hash": "h1", "_sample_idx": 0,
        }
    ]
    judge_by_ck = {"ckX": {"vr_flag": False, "mfr_flag": False}}
    condition_meta = {
        "ckX": {
            "model": "m1", "family": "B", "length": 300,
            "target": "A", "target_letter": "A", "baseline_letter": "B",
            "item_hash": "h1",
        }
    }
    enriched = build_enriched_results(subject_results, judge_by_ck, {}, condition_meta)
    assert len(enriched) == 1
    r = enriched[0]
    assert r["baseline_letter"] == "B"
    assert r["target_letter"] == "A"
    assert r["capturable"] is True       # baseline B != target A
    assert r["cir"] == 1                 # boxed A == target A
    assert r["eemr_conj"] == 1           # captured AND monitor missed (mfr_flag False)
    # grouping uses the merged family/length
    summary = aggregate_results(enriched)
    cell = next(iter(summary["conditions"].values()))
    assert cell["family"] == "B" and cell["length"] == 300


def test_build_enriched_results_unresolved_target_not_capturable(tmp_path):
    """Empty target/baseline (no condition_meta) must not count as capturable."""
    subject_results = [
        {"boxed_letter": "A", "trace": "t", "trace_letter": "A",
         "dissociation": False, "_condition_key": "ck0", "_family": "B",
         "_length": 100, "_target": "", "_item_hash": "", "_sample_idx": 0},
    ]
    enriched = build_enriched_results(subject_results, {}, {}, {})
    assert enriched[0]["capturable"] is False
    assert enriched[0]["cir"] == 0


def test_aggregate_results_groups_correctly(tmp_path):
    """aggregate_results groups by (model, family, length, target, variant, state)."""
    from src.metrics.core import enrich

    def _r(boxed, mfr_flag=False):
        r = {
            "boxed_letter": boxed,
            "trace_letter": boxed,
            "baseline_letter": "B",
            "target_letter": "A",
            "vr_flag": False,
            "mfr_flag": mfr_flag,
            "dissociation": False,
            "model": "m1",
            "family": "A",
            "length": 100,
            "target": "A",
            "monitor_variant": "full_trace",
            "monitored_state": "unmonitored",
        }
        enrich(r)
        return r

    enriched = [_r("A"), _r("A", mfr_flag=True), _r("B")]
    summary = aggregate_results(enriched)
    assert summary["n_enriched"] == 3
    assert len(summary["conditions"]) == 1
    cell = next(iter(summary["conditions"].values()))
    assert cell["n_total"] == 3
    assert cell["n_capturable"] == 3
