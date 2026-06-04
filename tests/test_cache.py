"""Tests for Cache: round-trip, truncated-line survival, dry_run, hit/miss stats."""
import json
import tempfile
import pathlib

import pytest

from src.infra.cache import Cache, canonical_key


@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path


def test_canonical_key_stable():
    k1 = canonical_key(a=1, b="x", c=[1, 2])
    k2 = canonical_key(b="x", a=1, c=[1, 2])  # different insertion order
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_miss_writes_and_hit_returns(tmp_root):
    c = Cache(tmp_root, "test_ns")
    calls = []

    def compute():
        calls.append(1)
        return {"result": 42}

    k = c.key(x=1)
    v = c.get_or_compute(shard="s1", key=k, compute_fn=compute)
    assert v == {"result": 42}
    assert len(calls) == 1
    assert c.misses == 1
    assert c.hits == 0

    # Second call — should be a cache HIT, compute_fn not called again.
    v2 = c.get_or_compute(shard="s1", key=k, compute_fn=compute)
    assert v2 == {"result": 42}
    assert len(calls) == 1  # not called again
    assert c.hits == 1


def test_hit_on_fresh_instance(tmp_root):
    """Data written by one Cache instance must be readable by a new one (same shard file)."""
    k = canonical_key(x=99)
    c1 = Cache(tmp_root, "ns2")
    c1.get_or_compute(shard="s", key=k, compute_fn=lambda: "hello")

    c2 = Cache(tmp_root, "ns2")
    result = c2.get_or_compute(shard="s", key=k, compute_fn=lambda: "SHOULD_NOT_BE_CALLED")
    assert result == "hello"
    assert c2.hits == 1


def test_truncated_final_line_tolerated(tmp_root):
    """A truncated (garbled) last line on the shard file is skipped; earlier records readable."""
    c = Cache(tmp_root, "ns3")
    k_good = canonical_key(x=1)
    c.get_or_compute(shard="s", key=k_good, compute_fn=lambda: "good")

    # Corrupt the shard file by appending an incomplete line.
    shard_path = tmp_root / "cache" / "ns3" / "s.jsonl"
    with open(shard_path, "a") as f:
        f.write('{"key": "truncated", "result": "oops"')  # no closing brace or newline

    # New instance should load the good record and skip the truncated line.
    c2 = Cache(tmp_root, "ns3")
    result = c2.get_or_compute(shard="s", key=k_good, compute_fn=lambda: "SHOULD_NOT")
    assert result == "good"
    assert c2.hits == 1


def test_dry_run_does_not_call_compute(tmp_root):
    c = Cache(tmp_root, "dry_ns", dry_run=True)
    k = c.key(x=5)
    calls = []
    result = c.get_or_compute(shard="s", key=k, compute_fn=lambda: calls.append(1) or "v")
    assert result is None  # dry_run never returns real value
    assert len(calls) == 0
    # Also should not write to disk.
    shard_path = tmp_root / "cache" / "dry_ns" / "s.jsonl"
    assert not shard_path.exists()


def test_stats(tmp_root):
    c = Cache(tmp_root, "stats_ns")
    k1, k2 = c.key(x=1), c.key(x=2)
    c.get_or_compute(shard="s", key=k1, compute_fn=lambda: 1)
    c.get_or_compute(shard="s", key=k2, compute_fn=lambda: 2)
    c.get_or_compute(shard="s", key=k1, compute_fn=lambda: 999)  # hit
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 2
