"""Append-only per-run manifest (§6.6.4). One JSON file per run under <root>/ledger/.

Each run writes: config snapshot, matrix size, per-layer cache hit/miss, $ spent,
GPU-hours, errors, and timestamps — the audit trail that makes a result reconstructable.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def write_manifest(
    root: os.PathLike | str,
    *,
    cfg: dict,
    matrix_size: int,
    cache_stats: dict[str, Any],
    usd_spent: float = 0.0,
    gpu_hours: float = 0.0,
    errors: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """Append a manifest JSON to <root>/ledger/ and return its path."""
    ledger_dir = Path(root) / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    ts = time.time()
    ts_str = time.strftime("%Y%m%dT%H%M%S", time.gmtime(ts))
    path = ledger_dir / f"{ts_str}.json"

    manifest = {
        "ts": ts,
        "ts_human": ts_str,
        "config": cfg,
        "matrix_size": matrix_size,
        "cache_stats": cache_stats,
        "usd_spent": usd_spent,
        "gpu_hours": gpu_hours,
        "errors": errors or [],
        **(extra or {}),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return path
