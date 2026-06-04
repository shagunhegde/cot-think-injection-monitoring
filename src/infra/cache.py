"""Sharded, content-addressable, crash-tolerant cache (§6.6.2).

Every expensive operation is keyed by a sha256 of its exact inputs, so a repeat
request returns the cached value and running the same command twice does no extra
work. Records are appended to a per-shard JSONL on `<root>/cache/<namespace>/`
*immediately* after computation (durable across Colab timeouts). A truncated final
line (process killed mid-append) is tolerated on load.

Shard scheme (§6.6.1):
  baselines/{model}.jsonl
  targets/{model}.jsonl
  prefills/{dataset}_{family}_{length}.jsonl
  subject/{subject_model}/{family}_{length}.jsonl
  judge/{judge_model}_{variant}.jsonl
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional


def canonical_key(**inputs: Any) -> str:
    """Canonical sorted-JSON -> sha256 hex. Stable across runs/machines."""
    blob = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """One namespace = one subtree under <root>/cache/. One shard file per logical
    group so files stay small and writes don't contend. Loaded lazily per shard."""

    def __init__(self, root: os.PathLike | str, namespace: str, dry_run: bool = False):
        self.base = Path(root) / "cache" / namespace
        self.base.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self._shards: dict[str, dict[str, dict]] = {}
        self.hits = 0
        self.misses = 0

    # -- key construction -------------------------------------------------
    def key(self, **inputs: Any) -> str:
        return canonical_key(**inputs)

    # -- shard io ---------------------------------------------------------
    def _shard_path(self, shard: str) -> Path:
        return self.base / f"{shard}.jsonl"

    def _load_shard(self, shard: str) -> dict[str, dict]:
        if shard in self._shards:
            return self._shards[shard]
        records: dict[str, dict] = {}
        path = self._shard_path(shard)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Truncated/garbled final line from a crash mid-append.
                        # Skip it; the entry will be recomputed on miss.
                        continue
                    if isinstance(rec, dict) and "key" in rec:
                        records[rec["key"]] = rec
        self._shards[shard] = records
        return records

    def _append(self, shard: str, rec: dict) -> None:
        path = self._shard_path(shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # -- public api -------------------------------------------------------
    def contains(self, shard: str, key: str) -> bool:
        return key in self._load_shard(shard)

    def get(self, shard: str, key: str) -> Optional[Any]:
        rec = self._load_shard(shard).get(key)
        return rec["result"] if rec else None

    def get_or_compute(
        self,
        *,
        shard: str,
        key: str,
        compute_fn: Callable[[], Any],
        meta: Optional[dict] = None,
    ) -> Any:
        records = self._load_shard(shard)
        if key in records:
            self.hits += 1
            return records[key]["result"]

        self.misses += 1
        if self.dry_run:
            # Estimate-only: never call compute_fn, never write. Returns None so the
            # caller can count net-new work without spending.
            return None

        result = compute_fn()
        rec = {"key": key, "meta": meta or {}, "result": result, "ts": time.time()}
        self._append(shard, rec)
        records[key] = rec
        return result

    def stats(self) -> dict:
        per_shard = {s: len(d) for s, d in self._shards.items()}
        return {
            "namespace": str(self.base),
            "hits": self.hits,
            "misses": self.misses,
            "loaded_shards": per_shard,
        }
