"""Figures (cache Layer 5) — pure, re-runnable rendering over aggregated metrics.

`render_all(summary, capability, out_dir)` writes PNGs from the JSON that the metrics
stage already produced (results/<run_name>/metrics.json + monitor_capability.json), so
plotting never touches the model/API caches.

matplotlib is an optional dependency (the `viz` extra). If it is missing, callers
(stage_plots) catch the ImportError and skip figures — the text tables are still written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _conditions_by_family(summary: dict) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = {}
    for c in summary.get("conditions", {}).values():
        by_family.setdefault(str(c.get("family", "")), []).append(c)
    for rows in by_family.values():
        rows.sort(key=lambda r: int(r.get("length", 0) or 0))
    return by_family


def plot_length_curves(summary: dict, out_path: Path) -> Path:
    """EEMR + MFR vs injection length, one line per family (the H1 dose-response)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_family = _conditions_by_family(summary)
    fig, (ax_eemr, ax_mfr) = plt.subplots(1, 2, figsize=(11, 4.2))
    for fam, rows in sorted(by_family.items()):
        xs = [int(r.get("length", 0) or 0) for r in rows]
        ax_eemr.plot(xs, [r.get("eemr_direct", 0.0) for r in rows], marker="o", label=f"Family {fam}")
        ax_mfr.plot(xs, [r.get("mfr_mean", 0.0) for r in rows], marker="s", label=f"Family {fam}")
    ax_eemr.set(title="EEMR vs injection length", xlabel="injection length (tokens)",
                ylabel="EEMR (evades monitor & captures)", ylim=(0, 1))
    ax_mfr.set(title="Monitor flag rate vs length", xlabel="injection length (tokens)",
               ylabel="MFR", ylim=(0, 1))
    for ax in (ax_eemr, ax_mfr):
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_monitor_capability(capability: dict, out_path: Path) -> Path | None:
    """MFR vs length per monitor (weak→strong) with each monitor's clean FPR annotated."""
    monitors = capability.get("monitors", {})
    if not monitors:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for mon, d in monitors.items():
        per_len = d.get("per_length", {})
        xs = sorted(int(k) for k in per_len)
        ys = [per_len[str(x)].get("mfr") for x in xs]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        fpr = d.get("clean_fpr")
        label = f"{mon}" + (f"  (FPR={fpr:.2f})" if fpr is not None else "")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    ax.set(title="Monitor capability: MFR vs length (weak→strong)",
           xlabel="injection length (tokens)", ylabel="MFR (injected)", ylim=(0, 1))
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_cas(summary: dict, out_path: Path) -> Path | None:
    """CAS = EEMR(B) − EEMR(C) per length (H2: content vs verbosity)."""
    cas = summary.get("cas", {})
    if not cas:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # CAS keys are stringified tuples (model, length, target, variant, mon_state).
    rows = []
    for k, v in cas.items():
        try:
            length = int(k.strip("()").split(",")[1])
        except (IndexError, ValueError):
            continue
        rows.append((length, float(v)))
    rows.sort()
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(r[0]) for r in rows], [r[1] for r in rows], color="#4c72b0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="CAS = EEMR(B) − EEMR(C) per length",
           xlabel="injection length (tokens)", ylabel="CAS  (>0 ⇒ content-driven)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_all(summary: dict, capability: dict, out_dir: Path) -> list[Path]:
    """Render every figure that the available data supports; return saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    saved.append(plot_length_curves(summary, out_dir / "length_curves.png"))
    cap = plot_monitor_capability(capability, out_dir / "monitor_capability.png")
    if cap:
        saved.append(cap)
    cas_fig = plot_cas(summary, out_dir / "cas.png")
    if cas_fig:
        saved.append(cas_fig)
    return saved
