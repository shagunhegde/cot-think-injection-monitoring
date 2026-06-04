"""Net-new cost estimator and spend guardrail (§6.6.2).

`estimate(matrix, caches, cost_cfg)` counts only work NOT already in cache,
returning per-layer net-new counts, dollars, and GPU-hours. The runner calls
`can_proceed(report, accumulated_usd, max_spend_usd)` before each generation
and halts+checkpoints if the next op would breach the cap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Report:
    """Cost estimate for a matrix, separated by cache layer."""

    # Net-new counts (cache misses)
    baselines_new: int = 0          # Layer 1
    prefills_new: int = 0           # Layer 2 (model-agnostic)
    subject_gens_new: int = 0       # Layer 3 (most expensive)
    judge_calls_new: int = 0        # Layer 4

    # Already-cached savings
    baselines_cached: int = 0
    prefills_cached: int = 0
    subject_gens_cached: int = 0
    judge_calls_cached: int = 0

    # Cost estimates
    api_usd: float = 0.0
    gpu_hours: float = 0.0

    dry_run: bool = False
    notes: list[str] = field(default_factory=list)

    def total_usd(self) -> float:
        return self.api_usd

    def is_zero(self) -> bool:
        return (
            self.baselines_new == 0
            and self.prefills_new == 0
            and self.subject_gens_new == 0
            and self.judge_calls_new == 0
        )

    def summary(self) -> str:
        lines = [
            "── Cost estimate ──────────────────────────────",
            f"  Layer 1 baselines :  {self.baselines_new:>6} new / {self.baselines_cached} cached",
            f"  Layer 2 prefills  :  {self.prefills_new:>6} new / {self.prefills_cached} cached",
            f"  Layer 3 subject   :  {self.subject_gens_new:>6} new / {self.subject_gens_cached} cached",
            f"  Layer 4 judges    :  {self.judge_calls_new:>6} new / {self.judge_calls_cached} cached",
            f"  API cost (est.)   :  ${self.api_usd:.4f}",
            f"  GPU time (est.)   :  {self.gpu_hours:.2f} h",
        ]
        if self.dry_run:
            lines.append("  [DRY RUN — no work performed]")
        for note in self.notes:
            lines.append(f"  ⚠ {note}")
        lines.append("────────────────────────────────────────────────")
        return "\n".join(lines)


def estimate(
    matrix: list[dict],
    caches: dict[str, Any],       # {"baselines": Cache, "prefills": Cache, "subject": Cache, "judge": Cache}
    cost_cfg: dict,
) -> Report:
    """Count net-new work per layer and price it.

    `cost_cfg` keys:
      sec_per_gen: {model_id: float}          — GPU seconds per subject generation
      judge_avg_tokens: float                 — average tokens per judge/monitor call
      judge_price_per_1k: float               — USD per 1k tokens (Anthropic API)
      prefill_gen_avg_tokens: float           — average tokens per B/C/D generation
      prefill_gen_price_per_1k: float         — USD per 1k tokens
    """
    report = Report()
    sec_per_gen: dict = cost_cfg.get("sec_per_gen", {})
    judge_avg_tokens: float = float(cost_cfg.get("judge_avg_tokens", 500))
    judge_price_k: float = float(cost_cfg.get("judge_price_per_1k", 0.002))
    prefill_avg_tokens: float = float(cost_cfg.get("prefill_gen_avg_tokens", 800))
    prefill_price_k: float = float(cost_cfg.get("prefill_gen_price_per_1k", 0.002))

    seen_baselines: set[str] = set()
    seen_prefills: set[str] = set()

    baseline_cache = caches.get("baselines")
    prefill_cache = caches.get("prefills")
    subject_cache = caches.get("subject")
    judge_cache = caches.get("judge")

    for cond in matrix:
        model = cond.get("model", "")
        family = cond.get("family", "")
        length = cond.get("length", 0)
        item = cond.get("item", "")
        ck = cond.get("condition_key", "")

        # Layer 1: baseline per (model, item)
        bl_key = f"{model}|{item}"
        if bl_key not in seen_baselines:
            seen_baselines.add(bl_key)
            if baseline_cache and baseline_cache.contains("baselines", bl_key):
                report.baselines_cached += 1
            else:
                report.baselines_new += 1

        # Layer 2: prefill per (family, length, item) — model-agnostic
        pf_key = f"{family}|{length}|{item}"
        if pf_key not in seen_prefills:
            seen_prefills.add(pf_key)
            if prefill_cache and prefill_cache.contains("prefills", pf_key):
                report.prefills_cached += 1
            else:
                report.prefills_new += 1
                if family in ("B", "C", "D"):  # A is free template
                    report.api_usd += prefill_avg_tokens * prefill_price_k / 1000

        # Layer 3: subject generation per condition_key
        if subject_cache and subject_cache.contains("subject", ck):
            report.subject_gens_cached += 1
        else:
            report.subject_gens_new += 1
            secs = sec_per_gen.get(model, 10.0)
            report.gpu_hours += secs / 3600.0

        # Layer 4: judge + monitor calls per condition_key
        if judge_cache and judge_cache.contains("judge", ck):
            report.judge_calls_cached += 1
        else:
            report.judge_calls_new += 1
            # 2 calls per condition (faithfulness + monitor)
            report.api_usd += 2 * judge_avg_tokens * judge_price_k / 1000

    return report


def can_proceed(report: Report, accumulated_usd: float, max_spend_usd: float | None) -> bool:
    """Return False if executing the next batch would exceed the spend cap."""
    if max_spend_usd is None:
        return True
    return accumulated_usd + report.total_usd() <= max_spend_usd
