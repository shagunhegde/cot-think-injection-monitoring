"""The flexibility layer (§6.6.2): every "reduce runs" knob lives here.

`build_condition_matrix(cfg, items)` produces the full list of work units — one dict
per (dataset, item, model, family, length, target, monitor_variant, monitored_state,
sample_idx) — after applying every filter/cap from config. This is the single place
that decides how much work a run does, so run volume is *entirely* config-driven.

Each condition dict carries a stable `condition_key` (used to derive the per-sample
generation seed and as the spine of the Layer-3 cache key).
"""
from __future__ import annotations

import random
from typing import Any, Iterable, Optional

from .cache import canonical_key

# Family-specific valid lengths (§5.4). 0 = control/baseline, handled separately.
FAMILY_MAX_LENGTH = {"A": 700}          # A capped at 700 (1500-token conclusion is unnatural)
FAMILY_MIN_LENGTH = {"D": 300}          # D needs room to vacillate; no 100
ALL_FAMILIES = ("A", "B", "C", "D")


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def valid_family_length(family: str, length: int) -> bool:
    if length <= 0:
        return False  # control handled by the baseline layer, not the attack matrix
    if family in FAMILY_MAX_LENGTH and length > FAMILY_MAX_LENGTH[family]:
        return False
    if family in FAMILY_MIN_LENGTH and length < FAMILY_MIN_LENGTH[family]:
        return False
    return True


def _targets_for(cfg_targets: Any) -> list[str]:
    """Resolve the abstract target axis. Actual letters are bound later (data layer);
    here we only enumerate *slots* so the matrix is structural and model-agnostic."""
    if cfg_targets == "most_plausible" or cfg_targets is None:
        return ["most_plausible"]
    if cfg_targets == "sweep":
        return ["wrong_0", "wrong_1", "wrong_2"]
    return _as_list(cfg_targets)  # explicit, e.g. ["A", "C"]


def condition_key(cond: dict) -> str:
    """Stable identity of a work unit (excludes nothing that changes the generation)."""
    return canonical_key(
        dataset=cond["dataset"],
        item=cond["item"],
        model=cond["model"],
        family=cond["family"],
        length=cond["length"],
        target=cond["target"],
        monitor_variant=cond["monitor_variant"],
        monitored_state=cond["monitored_state"],
        sample_idx=cond["sample_idx"],
    )


def shard_conditions(conditions: list[dict], shard: int, total_shards: int) -> list[dict]:
    """Deterministically partition a condition list across parallel GPU workers.

    Worker `shard` (0-based) of `total_shards` gets every condition whose stable
    position in the condition_key-sorted order satisfies `i % total_shards == shard`.
    Sorting by condition_key first makes the partition independent of matrix build
    order, so `--shard 0..N-1` cover the matrix disjointly and exhaustively regardless
    of how the caller assembled `conditions`.
    """
    if total_shards <= 1:
        return conditions
    if not (0 <= shard < total_shards):
        raise ValueError(f"shard {shard} out of range for total_shards {total_shards}")
    ordered = sorted(conditions, key=lambda c: c["condition_key"])
    return [c for i, c in enumerate(ordered) if i % total_shards == shard]


def build_condition_matrix(cfg: dict, items: Optional[Iterable[Any]] = None) -> list[dict]:
    """Cartesian product over all axes, filtered/capped by every config knob.

    `items` is the list of item identifiers (hashes/ids) for the datasets; if omitted,
    placeholder ids item_0..item_{n_items-1} are used (lets the matrix be built and
    cost-estimated before any dataset is downloaded).
    """
    datasets = _as_list(cfg.get("datasets", ["mmlu"]))
    models = _as_list(cfg.get("models", []))
    families = _as_list(cfg.get("families", list(ALL_FAMILIES)))
    lengths = _as_list(cfg.get("lengths", []))
    monitor_variants = _as_list(cfg.get("monitor_variants", ["full_trace"]))
    monitored_states = _as_list(cfg.get("monitored_states", ["unmonitored"]))
    targets = _targets_for(cfg.get("targets", "most_plausible"))
    n_items = cfg.get("n_items")
    samples = int(cfg.get("samples_per_condition", 1))
    sample_fraction = float(cfg.get("sample_fraction", 1.0))
    seed = int(cfg.get("seed", 0))
    max_conditions = cfg.get("max_conditions")
    limit = cfg.get("limit")

    if items is None:
        n = n_items if n_items is not None else 0
        item_list = [f"item_{i}" for i in range(n)]
    else:
        item_list = list(items)
        if n_items is not None:
            item_list = item_list[:n_items]

    matrix: list[dict] = []
    for dataset in datasets:
        for item in item_list:
            for model in models:
                for family in families:
                    for length in lengths:
                        if not valid_family_length(family, length):
                            continue
                        for target in targets:
                            for variant in monitor_variants:
                                for mon_state in monitored_states:
                                    for sample_idx in range(samples):
                                        cond = {
                                            "dataset": dataset,
                                            "item": item,
                                            "model": model,
                                            "family": family,
                                            "length": length,
                                            "target": target,
                                            "monitor_variant": variant,
                                            "monitored_state": mon_state,
                                            "sample_idx": sample_idx,
                                        }
                                        cond["condition_key"] = condition_key(cond)
                                        matrix.append(cond)

    # Reproducible fractional subsample (seeded).
    if 0.0 < sample_fraction < 1.0 and matrix:
        rng = random.Random(seed)
        k = max(1, round(len(matrix) * sample_fraction))
        matrix = sorted(rng.sample(matrix, k), key=lambda c: c["condition_key"])

    # Hard caps (safety valves).
    if max_conditions is not None:
        matrix = matrix[: int(max_conditions)]
    if limit is not None:
        matrix = matrix[: int(limit)]

    return matrix
