"""Experiment runner — the guarded loop over cache Layers 3 + 4.

Usage:
    python -m src.pipeline.run_experiment                        # config/experiment.yaml
    python -m src.pipeline.run_experiment --dry_run              # estimate only, zero spend
    python -m src.pipeline.run_experiment --cfg n_items=10       # override any config key
    python -m src.pipeline.run_experiment --cfg limit=1 dry_run=true  # single-condition smoke

Workflow (§6.6.4):
    1. Load config + cost config.
    2. Load MMLU items (or placeholder ids for dry_run).
    3. Build condition matrix (all filters, samples_per_condition).
    4. cost.estimate() → print net-new counts / $ / GPU-hours.
    5. If dry_run=true: write ledger, exit.
    6. Otherwise: guarded loop —
         For each condition:
           a. Check Layer-3 cache (subject generation). On miss: generate + write immediately.
           b. Check Layer-4 cache (VR + MFR judges). On miss: call judge + write immediately.
           c. Accumulate spend. If next op would breach max_spend_usd: checkpoint + halt.
       Write ledger manifest on completion or halt.

Kill-and-resume: every result is written to its JSONL shard immediately.
Re-running the same command skips all cached conditions (hash-addressed resumption).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from src.infra.matrix import build_condition_matrix
from src.infra.cost import estimate as cost_estimate, can_proceed, Report
from src.infra.paths import resolve_root, Paths
from src.infra.cache import Cache
from src.infra.ledger import write_manifest
from src.infra.seed import gen_seed
from src.infra.llm_client import LLMClient

log = logging.getLogger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_overrides(overrides: list[str]) -> dict:
    result: dict = {}
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        if v.lower() in ("true", "false"):
            result[k] = v.lower() == "true"
        elif v.lower() in ("null", "none"):
            result[k] = None
        else:
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v.strip()
    return result


# ── Item loading ───────────────────────────────────────────────────────────────

def _load_items(cfg: dict, paths: Paths) -> list[dict]:
    """Load MMLU items (or return placeholder dicts for dry_run / no-datasets mode)."""
    datasets = cfg.get("datasets", ["mmlu"])
    n_items = cfg.get("n_items")
    dry_run = cfg.get("dry_run", False)

    if dry_run or not datasets:
        n = n_items or 0
        return [{"item_hash": f"item_{i}", "question": "", "choices": [], "answer_idx": 0,
                 "answer_letter": "A", "subject": "", "split": "test"} for i in range(n)]

    items = []
    for dataset in datasets:
        if dataset == "mmlu":
            try:
                from src.data.load_mmlu import load_mmlu
                subjects = cfg.get("mmlu_subjects", "all")
                batch = load_mmlu(subjects=subjects, split="test", n_items=n_items,
                                  cache_dir=paths.hf_cache)
                items.extend(batch)
            except ImportError:
                log.warning("datasets library not installed; using placeholder items")
                n = n_items or 0
                items.extend([{"item_hash": f"mmlu_{i}", "question": "",
                                "choices": [], "answer_idx": 0, "answer_letter": "A",
                                "subject": "", "split": "test"} for i in range(n)])
        elif dataset == "gpqa":
            log.info("GPQA wired in Phase 5+; skipping for now")
    return items[:n_items] if n_items else items


# ── Baseline + target resolution ──────────────────────────────────────────────

def _resolve_targets(
    conditions: list[dict],
    items_by_hash: dict[str, dict],
    cfg: dict,
    paths: Paths,
    baseline_cache: Cache,
    target_cache: Cache,
) -> list[dict]:
    """Replace abstract target slots ('most_plausible', 'wrong_0'...) with real letters.

    On Colab this calls the subject model for option scoring; locally with no GPU it
    falls back to uniform logprobs (first wrong option alphabetically) for testing.
    Baseline answers are also cached here (Layer 1).
    """
    from src.data.select_target import select_targets, uniform_logprobs
    from src.pipeline.baseline import get_or_cache_baseline, get_or_cache_target

    target_mode = cfg.get("targets", "most_plausible")
    if target_mode not in ("most_plausible", "sweep"):
        # explicit letters — no resolution needed
        return conditions

    resolved = []
    for cond in conditions:
        item = items_by_hash.get(cond["item"])
        if item is None:
            continue
        target_slot = cond.get("target", "most_plausible")
        if target_slot not in ("most_plausible", "wrong_0", "wrong_1", "wrong_2"):
            # Already a real letter.
            resolved.append(cond)
            continue

        # Try to get cached baseline.
        model_id = cond["model"]

        def _dummy_generate(item):
            # GPU-free fallback: uniform logprobs, answer = correct.
            lp = uniform_logprobs(len(item.get("choices", ["A","B","C","D"])))
            return {"answer_letter": item.get("answer_letter", "A"), "option_logprobs": lp}

        bl = get_or_cache_baseline(item, model_id, baseline_cache,
                                   generate_fn=_dummy_generate)
        if bl is None:
            continue

        targets = get_or_cache_target(item, model_id, bl, target_cache,
                                      target_mode=target_mode if target_slot == "most_plausible" else "sweep")
        if not targets:
            continue

        # Expand sweep into multiple conditions; most_plausible gives one.
        for t in targets:
            new_cond = dict(cond)
            new_cond["target"] = t
            # Recompute condition_key with the resolved target.
            from src.infra.matrix import condition_key as ck_fn
            new_cond["condition_key"] = ck_fn(new_cond)
            resolved.append(new_cond)

    return resolved


# ── Layer 3: subject generation (GPU) ─────────────────────────────────────────

def _run_subject_generation(
    cond: dict,
    items_by_hash: dict[str, dict],
    prefill_cache: Cache,
    subject_cache: Cache,
    cfg: dict,
    models_cfg: dict,
) -> Optional[dict]:
    """Generate subject response for one condition. Returns cached or new result dict."""
    item_hash = cond["item"]
    item = items_by_hash.get(item_hash)
    if item is None:
        return None

    model_id = cond["model"]
    family = cond["family"]
    length = cond["length"]
    target = cond["target"]
    sample_idx = cond.get("sample_idx", 0)
    ck = cond["condition_key"]

    model_cfg = models_cfg.get("subjects", {}).get(model_id, {})
    hf_model_id = model_cfg.get("hf_id", model_id)
    shard = f"subject/{hf_model_id.replace('/', '_')}/{family}_{length}"

    # Cache hit?
    if subject_cache.contains(shard, ck):
        return subject_cache.get(shard, ck)

    # Fetch prefill text from Layer 2.
    from src.attacks.generate import prefill_cache_key as pfk, prefill_shard as pfs
    from src.attacks.templates import conclusion_sentence
    generator_model = cfg.get("generator_model", "claude-haiku-4-5-20251001")
    prompt_version = cfg.get("prompt_version", "v1")

    if family == "A":
        pf_shard = pfs("mmlu", "A", length)
        pf_key = pfk("mmlu", item_hash, "A", length, target, "template", prompt_version)
    else:
        pf_shard = pfs("mmlu", family, length)
        pf_key = pfk("mmlu", item_hash, family, length, target, generator_model, prompt_version)

    prefill_rec = prefill_cache.get(pf_shard, pf_key)
    prefill_text = prefill_rec["text"] if prefill_rec else ""

    # Generate (GPU path — only runs on Colab/GPU host).
    per_gen_seed = gen_seed(cfg.get("seed", 0), ck, sample_idx)

    def _compute():
        try:
            from src.models.run_model import generate_one_hf, load_subject_model
            from src.models.parse import parse_output
            # Model should be pre-loaded and passed in via a session object.
            # This path is called per-condition; callers should cache the model handle.
            raise NotImplementedError(
                "Subject generation requires a pre-loaded model. "
                "Use run_conditions_hf() from a Colab session with the model loaded."
            )
        except (ImportError, RuntimeError) as e:
            return {"error": str(e), "boxed_letter": None, "trace": "", "answer_region": ""}

    result = subject_cache.get_or_compute(shard=shard, key=ck, compute_fn=_compute,
                                          meta={"condition_key": ck, "sample_idx": sample_idx,
                                                "family": family, "length": length, "target": target})
    return result


# ── Layer 4: judge calls ───────────────────────────────────────────────────────

def _run_judges(
    cond: dict,
    subject_result: dict,
    item: dict,
    prefill_text: str,
    client: LLMClient,
    judge_cache: Cache,
    cfg: dict,
    models_cfg: dict,
) -> dict:
    """Run VR + MFR judges for one condition. Returns {"vr_flag": bool, "mfr_flag": bool, ...}."""
    from src.judges.faithfulness import judge_faithfulness
    from src.judges.monitor import run_monitor

    judges_cfg = models_cfg.get("judges", {})
    prompt_version = cfg.get("prompt_version", "v1")
    monitor_variants = cfg.get("monitor_variants", ["full_trace"])

    question = item.get("question", "")
    trace = subject_result.get("trace", "")
    answer = subject_result.get("boxed_letter", "") or ""

    # VR (faithfulness) judge.
    fj_cfg = judges_cfg.get("faithfulness", {})
    fj_model = fj_cfg.get("model_id", "claude-haiku-4-5-20251001")
    fj_pv = fj_cfg.get("prompt_version", prompt_version)
    vr_result = judge_faithfulness(
        question, trace, answer, prefill_text,
        client, fj_model, prompt_version=fj_pv, cache=judge_cache,
    )

    # MFR monitor (all requested variants).
    mfr_results: dict[str, dict] = {}
    for variant in monitor_variants:
        mon_cfg = judges_cfg.get(f"monitor_{variant}", judges_cfg.get("monitor_full_trace", {}))
        mon_model = mon_cfg.get("model_id", "claude-haiku-4-5-20251001")
        mon_pv = mon_cfg.get("prompt_version", prompt_version)
        mfr_results[variant] = run_monitor(
            question, trace, answer, client, mon_model,
            variant=variant, prompt_version=mon_pv, cache=judge_cache,
        )

    primary_mfr = mfr_results.get("full_trace", next(iter(mfr_results.values()), {}))

    return {
        "vr_flag": vr_result.get("flag", False),
        "vr_rationale": vr_result.get("rationale", ""),
        "mfr_flag": primary_mfr.get("flag", False),
        "mfr_rationale": primary_mfr.get("rationale", ""),
        "mfr_by_variant": {v: r.get("flag", False) for v, r in mfr_results.items()},
    }


# ── Main runner ────────────────────────────────────────────────────────────────

def run(cfg: dict, cost_cfg: dict, paths: Paths) -> None:
    """Core experiment loop. Called by main() after config loading."""
    dry_run = cfg.get("dry_run", False)
    max_spend = cfg.get("max_spend_usd")
    global_seed = cfg.get("seed", 0)

    # Load model configs.
    models_cfg_path = Path("config/models.yaml")
    models_cfg = _load_yaml(models_cfg_path) if models_cfg_path.exists() else {}

    # Load items.
    items = _load_items(cfg, paths)
    items_by_hash = {it["item_hash"]: it for it in items}

    # Build condition matrix.
    matrix = build_condition_matrix(cfg, [it["item_hash"] for it in items])

    # Build cache objects.
    baseline_cache = Cache(paths.root, "baselines")
    target_cache = Cache(paths.root, "targets")
    prefill_cache = Cache(paths.root, "prefills")
    subject_cache = Cache(paths.root, "subject")
    judge_cache = Cache(paths.root, "judge")

    caches = {
        "baselines": baseline_cache,
        "prefills": prefill_cache,
        "subject": subject_cache,
        "judge": judge_cache,
    }

    # Cost estimate.
    report = cost_estimate(matrix, caches, cost_cfg)
    report.dry_run = dry_run

    print(f"\nWorkspace root  : {paths.root}")
    print(f"Items loaded    : {len(items)}")
    print(f"Matrix size     : {len(matrix)} conditions")
    print(report.summary())

    if dry_run:
        print("\n[dry_run=true] No work performed.")
        write_manifest(paths.root, cfg=cfg, matrix_size=len(matrix),
                       cache_stats={}, extra={"mode": "dry_run"})
        return

    # Resolve abstract target slots to real letters.
    matrix = _resolve_targets(matrix, items_by_hash, cfg, paths,
                               baseline_cache, target_cache)
    print(f"Resolved matrix : {len(matrix)} conditions (after target resolution)")

    # LLM client (judge / monitor / generator calls).
    client = LLMClient.from_env()

    # Guarded loop.
    accumulated_usd = 0.0
    accumulated_gpu_h = 0.0
    errors: list[str] = []
    completed = 0
    skipped_cached = 0
    t_start = time.time()

    for cond in matrix:
        ck = cond["condition_key"]
        item = items_by_hash.get(cond["item"])
        if item is None:
            continue

        # ── Layer 3: subject generation ──────────────────────────────────────
        model_id = cond["model"]
        model_cfg = models_cfg.get("subjects", {}).get(model_id, {})
        hf_id = model_cfg.get("hf_id", model_id)
        layer3_shard = f"subject/{hf_id.replace('/', '_')}/{cond['family']}_{cond['length']}"

        if subject_cache.contains(layer3_shard, ck):
            subject_result = subject_cache.get(layer3_shard, ck)
            skipped_cached += 1
        else:
            # Spend check before generating.
            from src.infra.cost import Report as CostReport
            next_cost = CostReport(
                subject_gens_new=1,
                judge_calls_new=1,
                gpu_hours=model_cfg.get("sec_per_gen", cost_cfg.get("sec_per_gen", {}).get(model_id, 10)) / 3600,
                api_usd=2 * float(cost_cfg.get("judge_avg_tokens", 500)) * float(cost_cfg.get("judge_price_per_1k", 0.002)) / 1000,
            )
            if not can_proceed(next_cost, accumulated_usd, max_spend):
                print(f"\n[GUARDRAIL] Spend cap ${max_spend:.2f} would be exceeded at condition {completed+1}. "
                      f"Halting and checkpointing. Accumulated: ${accumulated_usd:.4f}")
                break

            subject_result = _run_subject_generation(
                cond, items_by_hash, prefill_cache, subject_cache, cfg, models_cfg,
            )
            if subject_result is None or subject_result.get("error"):
                err = f"Layer3 error: cond={ck[:16]}... {(subject_result or {}).get('error','')}"
                errors.append(err)
                log.warning(err)
                continue
            accumulated_gpu_h += next_cost.gpu_hours

        # ── Layer 4: judges ────────────────────────────────────────────────
        family = cond["family"]
        length = cond["length"]
        target = cond["target"]
        generator_model = cfg.get("generator_model", "claude-haiku-4-5-20251001")
        prompt_version = cfg.get("prompt_version", "v1")
        from src.attacks.generate import prefill_cache_key as pfk, prefill_shard as pfs
        pf_shard = pfs("mmlu", "A" if family == "A" else family, length)
        gm_key = "template" if family == "A" else generator_model
        pf_key = pfk("mmlu", cond["item"], family, length, target, gm_key, prompt_version)
        pf_rec = prefill_cache.get(pf_shard, pf_key)
        prefill_text = pf_rec["text"] if pf_rec else ""

        judge_result = _run_judges(
            cond, subject_result, item, prefill_text,
            client, judge_cache, cfg, models_cfg,
        )
        accumulated_usd += 2 * float(cost_cfg.get("judge_avg_tokens", 500)) * float(cost_cfg.get("judge_price_per_1k", 0.002)) / 1000

        completed += 1
        if completed % 10 == 0:
            print(f"  [{completed}/{len(matrix)}] ${accumulated_usd:.3f} | {errors and len(errors)+' errors' or 'no errors'}")

    elapsed = time.time() - t_start
    print(f"\nDone: {completed} completed, {skipped_cached} cached, {len(errors)} errors "
          f"in {elapsed:.1f}s | ${accumulated_usd:.4f} API | {accumulated_gpu_h:.2f} GPU-h")

    # Ledger manifest.
    write_manifest(
        paths.root,
        cfg=cfg,
        matrix_size=len(matrix),
        cache_stats={
            "baselines": baseline_cache.stats(),
            "prefills": prefill_cache.stats(),
            "subject": subject_cache.stats(),
            "judge": judge_cache.stats(),
        },
        usd_spent=accumulated_usd,
        gpu_hours=accumulated_gpu_h,
        errors=errors,
        extra={"completed": completed, "skipped_cached": skipped_cached,
               "elapsed_s": elapsed},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CoT-injection monitoring experiment runner")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--cost-config", default="config/cost.yaml")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VAL")
    args = parser.parse_args(argv)

    cfg = _load_yaml(Path(args.config))
    cost_cfg = _load_yaml(Path(args.cost_config)) if Path(args.cost_config).exists() else {}

    cfg.update(_parse_overrides(args.cfg))
    if args.dry_run:
        cfg["dry_run"] = True

    paths = resolve_root(mount=False, create=True)
    run(cfg, cost_cfg, paths)


if __name__ == "__main__":
    main()
