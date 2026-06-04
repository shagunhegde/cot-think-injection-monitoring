# CoT-Injection Monitoring

Measures how robust CoT-based oversight is when an attacker controls part of a model's `<think>` trace. Three pre-registered hypotheses:

- **H1** — monitorability degrades non-linearly with injection length
- **H2** — degradation is content-driven, not mere verbosity (CAS = EEMR(B) − EEMR(C))

---

## Setup

Requires Python 3.10+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync --extra openrouter --extra dev   # local dev (no GPU)
cp .env.example .env                     # add OPENROUTER_API_KEY (and HF_TOKEN)
```

For GPU hosts (Colab / RunPod):

```bash
uv sync --extra gpu --extra openrouter --extra dev
```

---

## Usage

Everything is configured via YAML files in `experiments/`. See `experiments/example_h2.yaml` for the annotated schema.

### Two-phase pipeline

**Phase 1 — Data generation** (GPU + attacker API):

```bash
# Always dry-run first to check cost + matrix size:
bash scripts/run_datagen.sh experiments/example_h2.yaml --dry-run

# Run all datagen stages (baselines → prefills → subject → clean_baseline):
bash scripts/run_datagen.sh experiments/example_h2.yaml all_datagen

# Or run a single stage:
bash scripts/run_datagen.sh experiments/example_h2.yaml subject
bash scripts/run_datagen.sh experiments/example_h2.yaml subject --shard 0 --total-shards 4
```

**Phase 2 — Analysis** (judges + metrics + plots, cheap / re-runnable):

```bash
bash scripts/run_pipeline.sh experiments/example_h2.yaml
```

Results land in `results/<run_name>/metrics.json`, `monitor_capability.json`, and `figures/`.

### Toggle individual stages in the config

```yaml
faithfulness: {enabled: true}
monitor:      {enabled: true}
clean_fpr:    {enabled: false}   # skip if clean baseline not yet run
metrics:      {enabled: true}
plots:        {enabled: true}
```

### Hypothesis presets

Set `hypothesis: H1`, `H2`, or `H3` in the config to automatically apply the standard families/lengths for that hypothesis.

---

## Model parametrisation

All attacker, judge, and monitor models are [OpenRouter](https://openrouter.ai/models) model IDs — one `OPENROUTER_API_KEY` reaches any of them:

```yaml
attacker:
  model: anthropic/claude-haiku-4.5

faithfulness:
  model: anthropic/claude-haiku-4.5

monitor:
  models:
    - qwen/qwen-2.5-0.5b-instruct       # weak open
    - qwen/qwen-2.5-1.5b-instruct       # scale-matched open
    - openai/gpt-4o-mini                # mid-tier closed
    - anthropic/claude-haiku-4.5        # strong closed
```

Subject models (open-weight, run on GPU) are keys in `config/models.yaml`:

```yaml
subjects:
  - deepseek-r1-distill-qwen-1.5b
  - qwen3-4b
```

---

## Running on Colab

Open `colab_launch.ipynb`, set keys and config in Cell 2, then run top-to-bottom. `resolve_root()` auto-mounts Drive and sets the workspace so caches survive disconnects.

## Running on RunPod

```bash
# Single-GPU full run:
python scripts/run_remote.py experiments/example_h2.yaml --gpu RTX3090 --terminate

# 4-GPU parallel subject inference:
for i in 0 1 2 3; do
    python scripts/run_remote.py experiments/example_full.yaml \
        --stage subject --shard $i --total-shards 4 &
done
wait
python scripts/run_remote.py experiments/example_full.yaml --stage all_analysis
```

Requires `RUNPOD_API_KEY` in `.env` and an SSH key at `~/.ssh/id_rsa` (or `--ssh-key PATH`).

---

## 5-layer content-addressable cache

Every expensive operation writes immediately to a sharded JSONL cache (crash-safe). Reruns skip anything already cached by SHA-256 key.

| Layer | What | When to re-run |
|---|---|---|
| L1 | Baselines + targets | Only when item set changes |
| L2 | Attack prefills (A/B/C/D) | Shared across **all** subject models |
| L3 | Subject continuations (GPU) | New subject model, family, length, or sample |
| L4 | VR judge + monitor flags | Bump `prompt_version` in config → L4 only |
| L5 | Metrics / plots | Always free to re-run |

---

## Tests

```bash
pytest tests/ -v
```

All tests run locally without GPU or API keys (`COTIM_LLM_BACKEND=mock` is set automatically).
