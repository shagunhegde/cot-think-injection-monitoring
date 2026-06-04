"""MMLU dataset loader (open access via cais/mmlu on HuggingFace).

Downloads once to <root>/data/mmlu_raw/ (via HF_DATASETS_CACHE); reads offline
thereafter. Returns a list of normalized item dicts with stable item_hash keys.

Schema per item:
    item_hash   : str  — sha256 of (dataset, subject, split, index)
    question    : str
    choices     : list[str]  — always 4 options
    answer_idx  : int  — correct answer index (0–3)
    answer_letter: str — correct answer letter (A–D)
    subject     : str
    split       : str
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _item_hash(dataset: str, subject: str, split: str, idx: int) -> str:
    blob = f"{dataset}|{subject}|{split}|{idx}".encode()
    return hashlib.sha256(blob).hexdigest()


def _normalize(row: dict, subject: str, split: str, idx: int) -> dict:
    answer_idx = int(row["answer"])
    return {
        "item_hash": _item_hash("mmlu", subject, split, idx),
        "question": row["question"],
        "choices": list(row["choices"]),
        "answer_idx": answer_idx,
        "answer_letter": IDX_TO_LETTER[answer_idx],
        "subject": subject,
        "split": split,
    }


def load_mmlu(
    subjects: list[str] | str = "all",
    split: str = "test",
    n_items: Optional[int] = None,
    cache_dir: Optional[os.PathLike] = None,
) -> list[dict]:
    """Load MMLU items. `subjects` can be a list or the string 'all'.

    `cache_dir` should point at <root>/data/hf_cache so weights survive Colab restarts.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets library required: pip install datasets") from e

    if isinstance(subjects, str) and subjects != "all":
        subjects = [subjects]

    kwargs: dict = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    items: list[dict] = []

    if subjects == "all" or subjects == ["all"]:
        ds = load_dataset("cais/mmlu", "all", split=split, **kwargs)
        for idx, row in enumerate(ds):
            items.append(_normalize(row, row.get("subject", "all"), split, idx))
            if n_items is not None and len(items) >= n_items:
                break
    else:
        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject, split=split, **kwargs)
            for idx, row in enumerate(ds):
                items.append(_normalize(row, subject, split, idx))
                if n_items is not None and len(items) >= n_items:
                    break
            if n_items is not None and len(items) >= n_items:
                break

    return items
