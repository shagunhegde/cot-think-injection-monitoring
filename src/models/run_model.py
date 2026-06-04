"""Subject-model generation — cache Layer 3 (§6.6.2).

Runs the subject model (DeepSeek-R1-Distill or Qwen3) on a batch of conditions,
writing each result to the cache immediately after generation.

Two backends:
  - HuggingFace Transformers (default, control path)
  - vLLM (throughput path; groups cache-misses into batches)

Per-generation seed: derived via gen_seed(global_seed, condition_key, sample_idx)
and passed to torch.manual_seed (HF) or SamplingParams(seed=...) (vLLM).

Engineering constraints (§7):
  - float16 for DeepSeek on T4; dtype=auto (bf16) for Qwen3.
  - add_special_tokens=False when tokenizing the already-templated string.
  - skip_special_tokens=False when decoding so </think> stays visible.
  - OOM guard: halve max_new_tokens/batch once on CUDA OOM; per-item try/except.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from src.infra.cache import Cache, canonical_key
from src.infra.seed import gen_seed
from src.infra.retry import oom_guard
from src.models.prefill import build_prefill_prompt, build_clean_prompt
from src.models.parse import parse_output


def _subject_cache_key(condition: dict) -> str:
    return condition["condition_key"]


def load_subject_model(model_cfg: dict, *, device: str = "cuda"):
    """Load an HF subject model. Call once per session; cache the return value.

    model_cfg keys (from config/models.yaml subjects section):
      hf_id, dtype, max_new_tokens, system_prompt (optional)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = model_cfg["hf_id"]
    dtype_str = model_cfg.get("dtype", "float16")

    # Resolve dtype string → torch dtype.
    import torch
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if dtype == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, device_map=device, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )
    model.eval()
    return model, tokenizer


def generate_one_hf(
    item: dict,
    prefill_text: str,
    model,
    tokenizer,
    model_cfg: dict,
    *,
    seed: int,
    device: str = "cuda",
    print_tail: bool = True,
) -> dict:
    """Generate a single response (HF Transformers path). Returns parse_output dict."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    system_prompt = model_cfg.get("system_prompt")
    max_new_tokens = model_cfg.get("max_new_tokens", 2048)
    temperature = model_cfg.get("temperature", 0.6)
    top_p = model_cfg.get("top_p", 0.95)
    top_k = model_cfg.get("top_k", None)

    prompt = build_prefill_prompt(
        item, prefill_text, tokenizer,
        system_prompt=system_prompt, print_tail=print_tail,
    )

    # add_special_tokens=False: the string is already templated — avoid double-BOS.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    gen_kwargs: dict[str, Any] = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k is not None:
        gen_kwargs["top_k"] = top_k

    def _gen(max_new_tokens, batch_size=1, **kw):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **kw)
        # Decode only the newly generated tokens.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        # skip_special_tokens=False so </think> stays visible.
        return tokenizer.decode(new_tokens, skip_special_tokens=False)

    full_output = oom_guard(_gen, max_new_tokens=max_new_tokens, **gen_kwargs)
    return parse_output(full_output)


def run_conditions_hf(
    conditions: list[dict],
    items_by_hash: dict[str, dict],
    prefills_by_key: dict[str, str],
    model,
    tokenizer,
    model_cfg: dict,
    cache: Cache,
    *,
    global_seed: int = 0,
    device: str = "cuda",
) -> list[dict]:
    """Run subject generation for a list of conditions (HF path).

    Each condition that is a cache miss is generated and written immediately.
    Conditions that are cache hits are skipped (idempotent + resumable).

    Returns the list of results (one per condition, in order).
    """
    results = []
    for cond in conditions:
        ck = cond["condition_key"]
        item_hash = cond["item"]
        item = items_by_hash.get(item_hash)
        if item is None:
            results.append({"condition_key": ck, "error": "item_not_found"})
            continue

        prefill_key = f"{cond['family']}|{cond['length']}|{item_hash}"
        prefill_text = prefills_by_key.get(prefill_key, "")

        per_gen_seed = gen_seed(global_seed, ck, cond.get("sample_idx", 0))

        def compute(item=item, prefill_text=prefill_text, seed=per_gen_seed):
            try:
                return generate_one_hf(
                    item, prefill_text, model, tokenizer, model_cfg,
                    seed=seed, device=device, print_tail=False,
                )
            except RuntimeError as e:
                return {"error": str(e)}

        result = cache.get_or_compute(
            shard=f"subject/{model_cfg['hf_id'].replace('/', '_')}/{cond['family']}_{cond['length']}",
            key=ck,
            compute_fn=compute,
            # Self-describing meta so Layer-5 aggregation can group/join without
            # rebuilding the matrix (item_hash/family/length/target carried here).
            meta={
                "condition_key": ck,
                "sample_idx": cond.get("sample_idx", 0),
                "item_hash": item_hash,
                "family": cond["family"],
                "length": cond["length"],
                "target": cond.get("target", ""),
            },
        )
        results.append({"condition_key": ck, "result": result})

    return results
