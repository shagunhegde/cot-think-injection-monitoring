"""Stage orchestrator — the config-driven harness entrypoint.

Mirrors the Reasoning-Theater idiom: one YAML config per run (under experiments/),
per-stage `enabled:` toggles, and a single dispatcher that runs one stage (or all
enabled stages in a phase). Every stage is a thin wrapper over existing code and maps
to one cache layer, so reruns recompute nothing already cached.

Phases & stages
---------------
datagen  (GPU + attacker API):
    baselines       L1  resolve target letters + cache baseline answers
    prefills        L2  generate Family A/B/C/D injections via the attacker (OpenRouter)
    subject         L3  subject-model continuation of the injected <think> (GPU, shardable)
    clean_baseline  L3  subject-model clean (no-injection) answers (GPU, shardable)

analysis (judges + metrics, cheap / re-runnable):
    faithfulness    L4  VR judge (knows the plant)
    monitor         L4  blind MFR monitor SWEEP across cfg.monitor.models × variants
    clean_fpr       L4  monitor flag rate on clean traces (false-positive rate)
    metrics         L5  CIR/AKR/VR/MFR/EEMR/CAS/MCP + monitor-capability table
    plots           L5  length curves, monitor-capability, CAS figures

CLI
---
    python -m src.pipeline.stages --config experiments/example_h2.yaml --stage subject \
        --shard 0 --total-shards 4
    python -m src.pipeline.stages --config experiments/example_h2.yaml --stage all_analysis
    python -m src.pipeline.stages --config experiments/example_h2.yaml --dry-run

The shell scripts (scripts/run_datagen.sh, scripts/run_pipeline.sh) read the same YAML,
gate each stage on its `enabled:` flag, and call this module once per stage.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml

from src.infra.paths import resolve_root, build_paths, Paths
from src.infra.cache import Cache
from src.infra.matrix import build_condition_matrix, shard_conditions
from src.infra.cost import estimate as cost_estimate
from src.infra.ledger import write_manifest
from src.infra.seed import gen_seed

from src.pipeline.run_experiment import _load_items, _resolve_targets, _parse_overrides

log = logging.getLogger(__name__)

# Hypothesis presets — set families + lengths in one knob (CLAUDE.md cheap-run recipes).
HYPOTHESIS_PRESETS: dict[str, dict] = {
    "H1": {"families": ["B"], "lengths": [100, 300, 700]},
    "H2": {"families": ["B", "C"], "lengths": [100, 700]},
}

DATAGEN_STAGES = ["baselines", "prefills", "subject", "clean_baseline"]
ANALYSIS_STAGES = ["faithfulness", "monitor", "clean_fpr", "metrics", "plots"]
ALL_STAGES = DATAGEN_STAGES + ANALYSIS_STAGES

CLEAN_GEN_STAGES = {"baselines", "metrics", "plots"}  # need neither GPU nor API
API_STAGES = {"prefills", "faithfulness", "monitor", "clean_fpr"}
GPU_STAGES = {"subject", "clean_baseline"}


# ── Config loading ──────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run_config(config_path: str, overrides: Optional[list[str]] = None) -> dict:
    """Load an experiments/*.yaml (RT-style nested blocks) into a flat cfg.

    The flat cfg carries every key that build_condition_matrix / cost.estimate expect
    (datasets, models, families, lengths, ...), and the original nested blocks
    (run, attacker, monitor, faithfulness, per-stage `enabled:` ...) are preserved at the
    top level for the stage functions to read.
    """
    base = _load_yaml(Path("config/experiment.yaml"))  # flat defaults
    raw = _load_yaml(Path(config_path))                 # RT-style nested

    cfg: dict = dict(base)
    # Preserve every nested block verbatim (run/data/subjects/attacker/attack/monitor/...).
    cfg.update(raw)

    run = raw.get("run", {})
    data = raw.get("data", {})
    attacker = raw.get("attacker", {})
    attack = raw.get("attack", {})
    monitor = raw.get("monitor", {})
    cost_block = raw.get("cost", {})

    # ── Flatten the blocks onto the keys the matrix/cost layer understands ──
    if "run_name" in run: cfg["run_name"] = run["run_name"]
    if "results_dir" in run: cfg["results_dir"] = run["results_dir"]
    if "seed" in run: cfg["seed"] = run["seed"]

    for k in ("datasets", "n_items", "mmlu_subjects"):
        if k in data: cfg[k] = data[k]

    if "subjects" in raw: cfg["models"] = raw["subjects"]
    if "model" in attacker: cfg["generator_model"] = attacker["model"]
    if "prompt_version" in attacker: cfg["prompt_version"] = attacker["prompt_version"]

    for k in ("families", "lengths", "targets", "samples_per_condition"):
        if k in attack: cfg[k] = attack[k]
    if "variants" in monitor: cfg["monitor_variants"] = monitor["variants"]

    if "dry_run" in cost_block: cfg["dry_run"] = cost_block["dry_run"]
    if "max_spend_usd" in cost_block: cfg["max_spend_usd"] = cost_block["max_spend_usd"]

    # ── Hypothesis preset overrides families+lengths (unless they were set explicitly) ──
    hyp = raw.get("hypothesis")
    if hyp:
        preset = HYPOTHESIS_PRESETS.get(str(hyp).upper())
        if preset is None:
            raise ValueError(f"Unknown hypothesis preset {hyp!r}; choose from {list(HYPOTHESIS_PRESETS)}")
        if "families" not in attack:
            cfg["families"] = preset["families"]
        if "lengths" not in attack:
            cfg["lengths"] = preset["lengths"]

    # ── CLI overrides win over everything ──
    if overrides:
        cfg.update(_parse_overrides(overrides))

    cfg.setdefault("run_name", "run")
    cfg.setdefault("results_dir", "results")
    return cfg


def _enabled(cfg: dict, stage: str) -> bool:
    block = cfg.get(stage)
    if isinstance(block, dict):
        return bool(block.get("enabled", True))
    return True


def results_root(cfg: dict) -> Path:
    root = Path(cfg.get("results_dir", "results")) / cfg.get("run_name", "run")
    root.mkdir(parents=True, exist_ok=True)
    return root


# ── Shared helpers ────────────────────────────────────────────────────────────

def _caches(paths: Paths) -> dict[str, Cache]:
    return {
        "baselines": Cache(paths.root, "baselines"),
        "targets": Cache(paths.root, "targets"),
        "prefills": Cache(paths.root, "prefills"),
        "subject": Cache(paths.root, "subject"),
        "judge": Cache(paths.root, "judge"),
        "clean_baseline": Cache(paths.root, "clean_baseline"),
    }


def _items_and_matrix(cfg: dict, paths: Paths, caches: dict[str, Cache]) -> tuple[list[dict], dict, list[dict]]:
    """Load items, build the matrix, and resolve abstract targets to real letters (L1)."""
    items = _load_items(cfg, paths)
    items_by_hash = {it["item_hash"]: it for it in items}
    raw = build_condition_matrix(cfg, [it["item_hash"] for it in items])
    matrix = _resolve_targets(raw, items_by_hash, cfg, paths,
                              caches["baselines"], caches["targets"])
    return items, items_by_hash, matrix


def _models_cfg() -> dict:
    return _load_yaml(Path("config/models.yaml"))


def _reference_tokenizer(cfg: dict, models_cfg: dict):
    """Return the reference tokenizer for length control.

    Loads the real HF tokenizer when transformers is available (GPU/datagen host),
    else falls back to the offline MockTokenizer so prefill generation still runs in
    CPU-only / test environments.
    """
    from src.attacks.length_control import MockTokenizer
    ref_key = cfg.get("reference_tokenizer") or (cfg.get("models") or ["mock"])[0]
    hf_id = models_cfg.get("subjects", {}).get(ref_key, {}).get("hf_id", ref_key)
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        log.warning("Reference tokenizer %s unavailable (%s); using MockTokenizer.", hf_id, e)
        return MockTokenizer()


def _prefill_text(prefill_cache: Cache, cond: dict, dataset: str,
                  generator_model: str, prompt_version: str) -> Optional[str]:
    """Fetch the Layer-2 prefill text for a condition (None if not generated)."""
    from src.attacks.generate import prefill_cache_key, prefill_shard
    family = cond["family"]
    gm = "template" if family == "A" else generator_model
    key = prefill_cache_key(dataset, cond["item"], family, cond["length"],
                            cond["target"], gm, prompt_version)
    rec = prefill_cache.get(prefill_shard(dataset, family, cond["length"]), key)
    return rec["text"] if (rec and rec.get("text")) else None


def _subject_result(subject_cache: Cache, cond: dict, hf_id: str) -> Optional[dict]:
    shard = f"subject/{hf_id.replace('/', '_')}/{cond['family']}_{cond['length']}"
    return subject_cache.get(shard, cond["condition_key"])


# ════════════════════════════════════════════════════════════════════════════════
# DATAGEN STAGES
# ════════════════════════════════════════════════════════════════════════════════

def stage_baselines(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L1: resolve target letters + cache baseline answers (done inside _resolve_targets)."""
    items, _, matrix = _items_and_matrix(cfg, paths, caches)
    print(f"[baselines] {len(items)} items → {len(matrix)} resolved conditions "
          f"(baselines + targets cached, Layer 1).")
    return {"items": len(items), "conditions": len(matrix)}


def stage_prefills(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L2: generate Family A/B/C/D injections via the attacker model (OpenRouter)."""
    from src.infra.llm_client import LLMClient
    from src.attacks.generate import generate_all_prefills

    _, items_by_hash, matrix = _items_and_matrix(cfg, paths, caches)
    matrix = shard_conditions(matrix, shard, total_shards)
    models_cfg = _models_cfg()
    tokenizer = _reference_tokenizer(cfg, models_cfg)
    client = client or LLMClient.from_env()

    stats = generate_all_prefills(
        matrix, items_by_hash, client, tokenizer=tokenizer, cache=caches["prefills"],
        dataset=(cfg.get("datasets") or ["mmlu"])[0],
        generator_model=cfg.get("generator_model", "anthropic/claude-haiku-4.5"),
        prompt_version=cfg.get("prompt_version", "v1"),
        tolerance=cfg.get("length_tolerance", 0.12),
    )
    print(f"[prefills] cached={stats.cached} generated={stats.generated} failed={stats.failed}")
    return {"cached": stats.cached, "generated": stats.generated, "failed": stats.failed}


def stage_subject(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L3 (GPU): subject continuation of the injected <think>. Sharded across workers."""
    from src.models.run_model import load_subject_model, run_conditions_hf

    _, items_by_hash, matrix = _items_and_matrix(cfg, paths, caches)
    matrix = shard_conditions(matrix, shard, total_shards)
    models_cfg = _models_cfg()
    dataset = (cfg.get("datasets") or ["mmlu"])[0]
    gen_model = cfg.get("generator_model", "anthropic/claude-haiku-4.5")
    pv = cfg.get("prompt_version", "v1")
    device = cfg.get("subject", {}).get("device", "cuda") if isinstance(cfg.get("subject"), dict) else "cuda"

    by_model: dict[str, list[dict]] = defaultdict(list)
    for cond in matrix:
        by_model[cond["model"]].append(cond)

    total_done = 0
    for model_key, conds in by_model.items():
        model_cfg = models_cfg.get("subjects", {}).get(model_key, {})
        if not model_cfg:
            log.warning("[subject] no models.yaml entry for %s; skipping.", model_key)
            continue
        print(f"[subject] loading {model_cfg['hf_id']} for {len(conds)} conditions "
              f"(shard {shard}/{total_shards}) ...")
        model, tokenizer = load_subject_model(model_cfg, device=device)
        prefills_by_key = {}
        for cond in conds:
            txt = _prefill_text(caches["prefills"], cond, dataset, gen_model, pv)
            if txt:
                prefills_by_key[f"{cond['family']}|{cond['length']}|{cond['item']}"] = txt
        results = run_conditions_hf(conds, items_by_hash, prefills_by_key,
                                    model, tokenizer, model_cfg, caches["subject"],
                                    global_seed=cfg.get("seed", 0), device=device)
        total_done += len(results)
        del model, tokenizer
        _free_gpu()
    print(f"[subject] {total_done} conditions generated/cached.")
    return {"conditions": total_done}


def stage_clean_baseline(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L3 (GPU): subject clean (no-injection) answers — the causal baseline for CIR/EEMR."""
    from src.models.run_model import load_subject_model, generate_one_hf

    items = _load_items(cfg, paths)
    models_cfg = _models_cfg()
    device = cfg.get("clean_baseline", {}).get("device", "cuda") if isinstance(cfg.get("clean_baseline"), dict) else "cuda"
    clean_cache = caches["clean_baseline"]

    # Shard items deterministically by item_hash position.
    items_sorted = sorted(items, key=lambda it: it["item_hash"])
    if total_shards > 1:
        items_sorted = [it for i, it in enumerate(items_sorted) if i % total_shards == shard]

    total = 0
    for model_key in (cfg.get("models") or []):
        model_cfg = models_cfg.get("subjects", {}).get(model_key, {})
        if not model_cfg:
            continue
        shard_name = f"clean_baseline/{model_key}"
        print(f"[clean_baseline] {model_cfg['hf_id']}: {len(items_sorted)} items "
              f"(shard {shard}/{total_shards}) ...")
        model, tokenizer = load_subject_model(model_cfg, device=device)
        for item in items_sorted:
            ih = item["item_hash"]
            if clean_cache.contains(shard_name, ih):
                continue
            s = gen_seed(cfg.get("seed", 0), f"clean|{ih}", 0)

            def compute(item=item, s=s):
                out = generate_one_hf(item, "", model, tokenizer, model_cfg,
                                      seed=s, device=device, print_tail=False)
                return {"clean_letter": out.get("boxed_letter"),
                        "trace": out.get("trace", ""),
                        "has_think": out.get("has_think_block", False),
                        "correct_letter": item["answer_letter"]}

            clean_cache.get_or_compute(shard=shard_name, key=ih, compute_fn=compute,
                                       meta={"item_hash": ih, "model": model_key})
            total += 1
        del model, tokenizer
        _free_gpu()
    print(f"[clean_baseline] {total} clean generations cached.")
    return {"clean": total}


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS STAGES
# ════════════════════════════════════════════════════════════════════════════════

def stage_faithfulness(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L4: VR faithfulness judge (knows the plant)."""
    from src.infra.llm_client import LLMClient
    from src.judges.faithfulness import judge_faithfulness

    _, items_by_hash, matrix = _items_and_matrix(cfg, paths, caches)
    matrix = shard_conditions(matrix, shard, total_shards)
    models_cfg = _models_cfg()
    dataset = (cfg.get("datasets") or ["mmlu"])[0]
    gen_model = cfg.get("generator_model", "anthropic/claude-haiku-4.5")
    pv = cfg.get("prompt_version", "v1")
    fcfg = cfg.get("faithfulness", {}) if isinstance(cfg.get("faithfulness"), dict) else {}
    judge_model = fcfg.get("model", gen_model)
    fpv = fcfg.get("prompt_version", pv)
    client = client or LLMClient.from_env()

    judged = skipped = 0
    for cond in matrix:
        hf_id = models_cfg.get("subjects", {}).get(cond["model"], {}).get("hf_id", cond["model"])
        s = _subject_result(caches["subject"], cond, hf_id)
        if not s or s.get("error"):
            skipped += 1; continue
        plant = _prefill_text(caches["prefills"], cond, dataset, gen_model, pv)
        if not plant:
            skipped += 1; continue
        item = items_by_hash[cond["item"]]
        judge_faithfulness(item["question"], s.get("trace", ""), s.get("boxed_letter") or "",
                           plant, client, judge_model, prompt_version=fpv, cache=caches["judge"])
        judged += 1
    print(f"[faithfulness] judged={judged} skipped={skipped} (judge={judge_model})")
    return {"judged": judged, "skipped": skipped}


def stage_monitor(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L4: blind monitor SWEEP across cfg.monitor.models × variants (weak→strong)."""
    from src.infra.llm_client import LLMClient
    from src.judges.monitor import run_monitor

    _, items_by_hash, matrix = _items_and_matrix(cfg, paths, caches)
    matrix = shard_conditions(matrix, shard, total_shards)
    models_cfg = _models_cfg()
    dataset = (cfg.get("datasets") or ["mmlu"])[0]
    gen_model = cfg.get("generator_model", "anthropic/claude-haiku-4.5")
    pv = cfg.get("prompt_version", "v1")

    mon_cfg = cfg.get("monitor", {}) if isinstance(cfg.get("monitor"), dict) else {}
    monitors = mon_cfg.get("models") or [gen_model]
    variants = mon_cfg.get("variants") or cfg.get("monitor_variants") or ["full_trace"]
    client = client or LLMClient.from_env()

    judged = skipped = 0
    for cond in matrix:
        hf_id = models_cfg.get("subjects", {}).get(cond["model"], {}).get("hf_id", cond["model"])
        s = _subject_result(caches["subject"], cond, hf_id)
        if not s or s.get("error"):
            skipped += 1; continue
        if not _prefill_text(caches["prefills"], cond, dataset, gen_model, pv):
            skipped += 1; continue
        item = items_by_hash[cond["item"]]
        q, tr, ans = item["question"], s.get("trace", ""), s.get("boxed_letter") or ""
        for monitor_model in monitors:
            for variant in variants:
                run_monitor(q, tr, ans, client, monitor_model, variant=variant,
                            prompt_version=pv, cache=caches["judge"])
                judged += 1
    print(f"[monitor] {judged} monitor calls over {len(monitors)} models × {len(variants)} "
          f"variants; skipped {skipped} conditions.")
    return {"calls": judged, "monitors": monitors, "variants": variants}


def stage_clean_fpr(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L4: monitor flag rate on clean traces (false-positive rate) for each monitor."""
    from src.infra.llm_client import LLMClient
    from src.judges.monitor import run_monitor

    items = _load_items(cfg, paths)
    items_by_hash = {it["item_hash"]: it for it in items}
    pv = cfg.get("prompt_version", "v1")
    gen_model = cfg.get("generator_model", "anthropic/claude-haiku-4.5")
    mon_cfg = cfg.get("monitor", {}) if isinstance(cfg.get("monitor"), dict) else {}
    monitors = mon_cfg.get("models") or [gen_model]
    variants = mon_cfg.get("variants") or cfg.get("monitor_variants") or ["full_trace"]
    client = client or LLMClient.from_env()
    clean_cache = caches["clean_baseline"]

    n = 0
    for model_key in (cfg.get("models") or []):
        shard_name = f"clean_baseline/{model_key}"
        for ih, item in items_by_hash.items():
            cr = clean_cache.get(shard_name, ih)
            if not cr or not cr.get("trace"):
                continue
            q, tr, ans = item["question"], cr["trace"], cr.get("clean_letter") or ""
            for monitor_model in monitors:
                for variant in variants:
                    run_monitor(q, tr, ans, client, monitor_model, variant=variant,
                                prompt_version=pv, cache=caches["judge"])
                    n += 1
    print(f"[clean_fpr] {n} monitor calls on clean traces.")
    return {"calls": n}


def stage_metrics(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L5: standard aggregate (CIR/AKR/VR/MFR/EEMR/CAS/MCP) + monitor-capability sweep table."""
    from src.pipeline.aggregate import (
        load_subject_results, load_judge_results, load_baseline_targets,
        build_condition_meta, build_enriched_results, aggregate_results,
        _print_length_table,
    )
    out_dir = results_root(cfg)

    items = _load_items(cfg, paths)
    items_by_hash = {it["item_hash"]: it for it in items}
    condition_meta = build_condition_meta(cfg, paths, items_by_hash,
                                          caches["baselines"], caches["targets"])
    subject_results = load_subject_results(paths.root)
    judge_by_ck = load_judge_results(paths.root)
    baseline_by_ck = load_baseline_targets(paths.root)

    if not subject_results:
        print("[metrics] no subject results found — run the datagen phase first.")
        return {"enriched": 0}

    enriched = build_enriched_results(subject_results, judge_by_ck, baseline_by_ck,
                                      condition_meta=condition_meta)
    summary = aggregate_results(enriched)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[metrics] wrote {metrics_path} ({summary['n_enriched']} enriched, "
          f"{len(summary['conditions'])} condition cells)")
    _print_length_table(summary)

    # Monitor-capability sweep table (per monitor × length: MFR + clean FPR).
    cap = monitor_capability_table(cfg, paths, caches, items_by_hash)
    cap_path = out_dir / "monitor_capability.json"
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(cap, f, indent=2, default=str)
    print(f"[metrics] wrote {cap_path}")
    _print_capability_table(cap)
    return {"enriched": summary["n_enriched"], "conditions": len(summary["conditions"])}


def stage_plots(cfg, paths, caches, *, shard=0, total_shards=1, client=None) -> dict:
    """L5: render figures from the aggregated metrics."""
    out_dir = results_root(cfg)
    metrics_path = out_dir / "metrics.json"
    cap_path = out_dir / "monitor_capability.json"
    if not metrics_path.exists():
        print("[plots] metrics.json missing — run the metrics stage first.")
        return {"figures": 0}
    summary = json.loads(metrics_path.read_text())
    cap = json.loads(cap_path.read_text()) if cap_path.exists() else {}
    try:
        from src.viz.plots import render_all
    except Exception as e:  # noqa: BLE001
        print(f"[plots] viz unavailable ({e}); skipping figures (text tables already written).")
        return {"figures": 0}
    figs = render_all(summary, cap, out_dir / "figures")
    print(f"[plots] wrote {len(figs)} figures to {out_dir / 'figures'}")
    return {"figures": len(figs)}


# ── Monitor-capability table (reads Layer-4 judge cache directly) ───────────────

def monitor_capability_table(cfg, paths, caches, items_by_hash) -> dict:
    """Per monitor model × length: injected MFR + parse rate, and clean-trace FPR.

    This is the weak→strong headline comparison. It reads the Layer-4 judge cache by
    monitor model so the sweep stays separable from the primary aggregate.
    """
    from src.judges.monitor import monitor_cache_key
    judge_cache = caches["judge"]
    clean_cache = caches["clean_baseline"]
    dataset = (cfg.get("datasets") or ["mmlu"])[0]
    gen_model = cfg.get("generator_model", "anthropic/claude-haiku-4.5")
    pv = cfg.get("prompt_version", "v1")
    mon_cfg = cfg.get("monitor", {}) if isinstance(cfg.get("monitor"), dict) else {}
    monitors = mon_cfg.get("models") or [gen_model]
    variant = (mon_cfg.get("variants") or cfg.get("monitor_variants") or ["full_trace"])[0]

    _, _, matrix = _items_and_matrix(cfg, paths, caches)
    models_cfg = _models_cfg()

    def lookup(monitor_model, q, tr, ans):
        key = monitor_cache_key(monitor_model, variant, pv, q, tr, ans)
        return judge_cache.get(f"judge/{monitor_model.replace('/', '_')}_{variant}", key)

    table: dict[str, Any] = {"variant": variant, "monitors": {}}
    for monitor_model in monitors:
        per_len = defaultdict(lambda: {"n": 0, "parsed": 0, "flag": 0})
        for cond in matrix:
            hf_id = models_cfg.get("subjects", {}).get(cond["model"], {}).get("hf_id", cond["model"])
            s = _subject_result(caches["subject"], cond, hf_id)
            if not s or s.get("error"):
                continue
            if not _prefill_text(caches["prefills"], cond, dataset, gen_model, pv):
                continue
            item = items_by_hash[cond["item"]]
            rec = lookup(monitor_model, item["question"], s.get("trace", ""),
                         s.get("boxed_letter") or "")
            d = per_len[cond["length"]]
            d["n"] += 1
            if rec:
                d["parsed"] += 1
                d["flag"] += int(bool(rec.get("flag")))
        # Clean FPR.
        fn = ff = 0
        for model_key in (cfg.get("models") or []):
            shard_name = f"clean_baseline/{model_key}"
            for ih, item in items_by_hash.items():
                cr = clean_cache.get(shard_name, ih)
                if not cr or not cr.get("trace"):
                    continue
                rec = lookup(monitor_model, item["question"], cr["trace"],
                             cr.get("clean_letter") or "")
                if rec:
                    fn += 1
                    ff += int(bool(rec.get("flag")))
        table["monitors"][monitor_model] = {
            "per_length": {str(ln): {
                "n": d["n"], "parsed": d["parsed"],
                "mfr": (d["flag"] / d["parsed"]) if d["parsed"] else None,
                "parse_rate": (d["parsed"] / d["n"]) if d["n"] else None,
            } for ln, d in sorted(per_len.items())},
            "clean_fpr": (ff / fn) if fn else None,
            "clean_n": fn,
        }
    return table


def _print_capability_table(cap: dict) -> None:
    mons = cap.get("monitors", {})
    if not mons:
        return
    print(f"\n{'monitor':>34} {'len':>5} {'n':>4} {'MFR':>7} {'parse%':>7} | {'FPR(clean)':>10}")
    print("-" * 78)
    for mon, d in mons.items():
        fpr = d.get("clean_fpr")
        fpr_s = f"{fpr:.3f} (n={d.get('clean_n', 0)})" if fpr is not None else "—"
        per_len = d.get("per_length", {})
        first = True
        for ln, row in per_len.items():
            mfr = row.get("mfr"); pr = row.get("parse_rate")
            mfr_s = f"{mfr:.3f}" if mfr is not None else "—"
            pr_s = f"{pr*100:.0f}%" if pr is not None else "—"
            print(f"{mon[:34]:>34} {ln:>5} {row['n']:>4} {mfr_s:>7} {pr_s:>7} | "
                  f"{(fpr_s if first else ''):>10}")
            first = False
        print("-" * 78)


# ── Misc ─────────────────────────────────────────────────────────────────────

def _free_gpu() -> None:
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


STAGE_FUNCS = {
    "baselines": stage_baselines,
    "prefills": stage_prefills,
    "subject": stage_subject,
    "clean_baseline": stage_clean_baseline,
    "faithfulness": stage_faithfulness,
    "monitor": stage_monitor,
    "clean_fpr": stage_clean_fpr,
    "metrics": stage_metrics,
    "plots": stage_plots,
}


def dry_run_report(cfg: dict, paths: Paths) -> None:
    """Print matrix size + per-layer net-new cost estimate (zero spend, no GPU)."""
    cost_cfg = _load_yaml(Path("config/cost.yaml"))
    caches = _caches(paths)
    # Use a dry-run copy so placeholder items don't trigger baseline/target resolution.
    dry_cfg = dict(cfg, dry_run=True)
    items = _load_items(dry_cfg, paths)
    items_by_hash = {it["item_hash"]: it for it in items}
    matrix = build_condition_matrix(cfg, [it["item_hash"] for it in items])
    report = cost_estimate(matrix, {k: caches[k] for k in ("baselines", "prefills", "subject", "judge")}, cost_cfg)
    report.dry_run = True
    print(f"\nWorkspace root  : {paths.root}")
    print(f"Run name        : {cfg.get('run_name')}")
    print(f"Items loaded    : {len(items)}")
    print(f"Matrix size     : {len(matrix)} conditions (before target resolution)")
    print(f"Subjects        : {cfg.get('models')}")
    print(f"Attacker        : {cfg.get('generator_model')}")
    mon = cfg.get("monitor", {})
    print(f"Monitor sweep   : {mon.get('models') if isinstance(mon, dict) else None}")
    print(report.summary())
    write_manifest(paths.root, cfg=cfg, matrix_size=len(matrix),
                   cache_stats={}, extra={"mode": "dry_run", "run_name": cfg.get("run_name")})


def run_stage(cfg: dict, paths: Paths, stage: str, *, shard=0, total_shards=1) -> dict:
    """Dispatch a single stage, or a phase pseudo-stage (all_datagen/all_analysis/all)."""
    if stage in ("all", "all_datagen", "all_analysis"):
        order = {"all": ALL_STAGES, "all_datagen": DATAGEN_STAGES,
                 "all_analysis": ANALYSIS_STAGES}[stage]
        results = {}
        for st in order:
            if not _enabled(cfg, st):
                print(f"[{st}] disabled — skipping.")
                continue
            results[st] = run_stage(cfg, paths, st, shard=shard, total_shards=total_shards)
        return results

    fn = STAGE_FUNCS.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage {stage!r}; choose from {list(STAGE_FUNCS)} "
                         f"or all/all_datagen/all_analysis.")
    caches = _caches(paths)
    t0 = time.time()
    out = fn(cfg, paths, caches, shard=shard, total_shards=total_shards)
    print(f"[{stage}] done in {time.time() - t0:.1f}s")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CoT-injection harness stage runner")
    parser.add_argument("--config", required=True, help="experiments/*.yaml run config")
    parser.add_argument("--stage", default="all",
                        help="one of " + ", ".join(STAGE_FUNCS) + ", or all/all_datagen/all_analysis")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="estimate cost + exit (zero spend)")
    parser.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VAL")
    parser.add_argument("--workspace", default=None, help="override workspace root")
    args = parser.parse_args(argv)

    cfg = load_run_config(args.config, args.cfg)
    if args.workspace:
        paths = build_paths(args.workspace, create=True)
    else:
        paths = resolve_root(mount=False, create=True)

    if args.dry_run or cfg.get("dry_run"):
        dry_run_report(cfg, paths)
        return

    run_stage(cfg, paths, args.stage, shard=args.shard, total_shards=args.total_shards)


if __name__ == "__main__":
    main()
