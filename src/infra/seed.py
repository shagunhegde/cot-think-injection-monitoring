"""Deterministic seed derivation (the reproducibility lever for `samples_per_condition`).

A single global `seed` in config plus the condition identity and the sample index
yield a unique, reproducible per-generation seed. Same config => byte-identical traces;
raising samples_per_condition only adds NEW sample indices (existing seeds unchanged).
"""
from __future__ import annotations

import hashlib

_MASK31 = (1 << 31) - 1  # torch / numpy accept seeds in [0, 2**31)


def gen_seed(global_seed: int, condition_key: str, sample_idx: int) -> int:
    """Stable int seed for one (condition, sample) generation."""
    blob = f"{global_seed}|{condition_key}|{sample_idx}".encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big") & _MASK31


def seed_everything(seed: int) -> None:
    """Best-effort global seeding for libraries that are present. No-op for absent ones."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
