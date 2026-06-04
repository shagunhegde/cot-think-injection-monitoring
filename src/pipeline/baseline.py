"""Cache Layer 1: baseline answers and target selection.

For each (model, item), cache:
  - The model's clean answer (no injection) — needed for CIR/AKR.
  - The selected target(s) — most-plausible distractor or sweep.

"Capture" is only defined when baseline ≠ target, so this must run before any
attack layer. Results are content-addressable under cache/baselines/{model}.jsonl
and cache/targets/{model}.jsonl.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from src.infra.cache import Cache, canonical_key
from src.data.select_target import select_targets, uniform_logprobs

IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _baseline_key(model_id: str, item_hash: str) -> str:
    return canonical_key(model_id=model_id, item_hash=item_hash)


def _target_key(model_id: str, item_hash: str, target_mode: str) -> str:
    return canonical_key(model_id=model_id, item_hash=item_hash, target_mode=target_mode)


def get_or_cache_baseline(
    item: dict,
    model_id: str,
    cache: Cache,
    *,
    generate_fn,   # fn(item) -> {"answer_letter": str, "option_logprobs": list[float]}
    dry_run: bool = False,
) -> Optional[dict]:
    """Return (and cache) the model's clean answer for one item.

    `generate_fn` should run a clean forward pass (no prefill) on the item and return:
      answer_letter   : str  — the model's answer (A–D)
      option_logprobs : list[float]  — log-probs over 4 options (for target selection)
    """
    key = _baseline_key(model_id, item["item_hash"])
    return cache.get_or_compute(
        shard=f"baselines/{model_id.replace('/', '_')}",
        key=key,
        compute_fn=lambda: generate_fn(item),
        meta={"model_id": model_id, "item_hash": item["item_hash"]},
    )


def get_or_cache_target(
    item: dict,
    model_id: str,
    baseline: dict,
    cache: Cache,
    *,
    target_mode: str = "most_plausible",
) -> Optional[list[str]]:
    """Return (and cache) the target letter(s) for one item, given its baseline."""
    key = _target_key(model_id, item["item_hash"], target_mode)

    def _compute():
        lp = baseline.get("option_logprobs") or uniform_logprobs(len(item["choices"]))
        return select_targets(item, lp, mode=target_mode)  # type: ignore[arg-type]

    return cache.get_or_compute(
        shard=f"targets/{model_id.replace('/', '_')}",
        key=key,
        compute_fn=_compute,
        meta={"model_id": model_id, "item_hash": item["item_hash"], "target_mode": target_mode},
    )


def run_baselines(
    items: list[dict],
    model_id: str,
    cache: Cache,
    *,
    generate_fn,
    target_mode: str = "most_plausible",
    dry_run: bool = False,
) -> list[dict]:
    """Run baseline + target selection for all items. Returns list of result dicts.

    Each result has:
      item_hash        : str
      model_id         : str
      baseline_letter  : str
      option_logprobs  : list[float]
      targets          : list[str]  — letter(s) to use as attack target
    """
    results = []
    for item in items:
        bl = get_or_cache_baseline(item, model_id, cache, generate_fn=generate_fn, dry_run=dry_run)
        if bl is None:
            # dry_run miss or compute returned None
            results.append({"item_hash": item["item_hash"], "model_id": model_id, "_dry_run_miss": True})
            continue

        targets = get_or_cache_target(item, model_id, bl, cache, target_mode=target_mode)
        results.append({
            "item_hash": item["item_hash"],
            "model_id": model_id,
            "baseline_letter": bl.get("answer_letter"),
            "option_logprobs": bl.get("option_logprobs"),
            "targets": targets,
        })
    return results
