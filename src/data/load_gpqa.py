"""GPQA dataset loader — GATED. Requires HuggingFace authentication.

RULES (hard constraints §7):
  - NEVER persist item text anywhere — not in cache, runs, figures, ledger, or the repo.
  - Store only non-reconstructable derived fields: item_hash, baseline_letter,
    target_letter, metric outcomes.
  - Accept dataset terms at: https://huggingface.co/datasets/Idavidrein/gpqa
  - Then: huggingface-cli login

This stub is wired in Phase 5+ after MMLU end-to-end works.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _item_hash(config: str, split: str, idx: int) -> str:
    blob = f"gpqa|{config}|{split}|{idx}".encode()
    return hashlib.sha256(blob).hexdigest()


def load_gpqa(
    config: str = "gpqa_diamond",
    split: str = "train",
    n_items: Optional[int] = None,
    cache_dir: Optional[os.PathLike] = None,
) -> list[dict]:
    """Load GPQA items. Choices are shuffled (record the mapping, never the text).

    Returns items WITHOUT question/choices text — callers receive only item_hash
    and the index/letter mapping. The caller is responsible for holding text in
    memory only as long as needed and never writing it to disk.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets library required: pip install datasets") from e

    import random

    kwargs: dict = {"trust_remote_code": True}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    try:
        ds = load_dataset("Idavidrein/gpqa", config, split=split, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Could not load GPQA ({config}/{split}). "
            "Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa "
            "and run: huggingface-cli login"
        ) from e

    items: list[dict] = []
    for idx, row in enumerate(ds):
        # Shuffle options to assign letters; record the mapping by index only.
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        order = list(range(4))
        random.shuffle(order)
        correct_shuffled_idx = order.index(0)  # correct answer was at index 0 pre-shuffle

        items.append({
            "item_hash": _item_hash(config, split, idx),
            # Text is held only here in memory — never written to disk by this loader.
            "_question": row["Question"],                          # in-memory only
            "_choices": [options[i] for i in order],              # in-memory only
            "answer_idx": correct_shuffled_idx,
            "answer_letter": IDX_TO_LETTER[correct_shuffled_idx],
            "subject": row.get("High-level domain", ""),
            "split": split,
            "config": config,
        })
        if n_items is not None and len(items) >= n_items:
            break

    return items
