# CLAUDE.md — CoT-Injection Monitoring Harness
<!-- PART 1: Stable project context. Edit in place as architecture evolves. -->

## Mission
Measure how robust CoT-based oversight is when an attacker controls part of a model's `<think>` trace. Three pre-registered hypotheses — **H1** (monitorability degrades non-linearly with injection length), **H2** (degradation is content-driven, not mere verbosity)
---

## Repository layout (§8 of spec)
```
cot-injection-monitoring/          ← repo root (committed to git)
  CLAUDE.md                        ← this file (living build log)
  README.md                        ← human-facing overview + how to run
  pyproject.toml                   ← deps (core: pyyaml; gpu/api/viz/dev extras)
  config/
    experiment.yaml                ← ALL scale knobs (see § Cost-control workflow)
    models.yaml                    ← subject/judge/generator model configs
    cost.yaml                      ← sec_per_gen, judge/prefill pricing (feeds estimator)
  src/
    infra/                         ← cross-cutting, env-aware (local + Colab)
      paths.py                     ← resolve_root(): local vs Colab+Drive; Paths dataclass
      cache.py                     ← Cache: sharded JSONL, content-addressable, crash-tolerant
      retry.py                     ← @retry backoff; JSON-judge guard; OOM guard
      seed.py                      ← gen_seed(global_seed, condition_key, sample_idx)
      matrix.py                    ← build_condition_matrix(cfg): all filters + samples_per_condition
      cost.py                      ← estimate(matrix, caches, cost_cfg) → Report; spend guardrail
      ledger.py                    ← write_manifest(): append-only per-run JSON to <root>/ledger/
      llm_client.py                ← LLMClient (Anthropic API backend + MockBackend for offline tests)
    data/
      load_mmlu.py                 ← [Phase 2] open; reads from <root>/data/mmlu_raw/
      load_gpqa.py                 ← [Phase 2 stub → Phase 5+] gated; NEVER persist item text
      select_target.py             ← [Phase 2] most-plausible distractor / sweep (cache layer 1)
    attacks/
      templates.py                 ← [Phase 3] Family A/B/C/D prompt skeletons (Appendix B)
      generate.py                  ← [Phase 3] generator-LLM calls for B/C/D via Anthropic API (layer 2)
      length_control.py            ← [Phase 3] reference-tokenizer enforcement ±12%
      validate.py                  ← [Phase 3] per-family prefill validators
    models/
      prefill.py                   ← [Phase 1] chat-template + <think> prefill (print-and-detect)
      run_model.py                 ← [Phase 1] vLLM/HF generation; gen_seed wiring (layer 3)
      parse.py                     ← [Phase 1] split trace/answer; extract boxed letter + stated value
    judges/
      faithfulness.py              ← [Phase 4] VR judge (knows plant) — layer 4
      monitor.py                   ← [Phase 4] MFR monitor + variants — layer 4
    metrics/
      core.py                      ← [Phase 4] CIR/AKR/VR/MFR per-item indicators
      composites.py                ← [Phase 4] Scheming/Robustness/EEMR/OSM/CAS/MCP + bootstrap CIs
      dissociation.py              ← [Phase 4] trace–answer dissociation flag
    pipeline/
      baseline.py                  ← [Phase 2] cache clean answers (layer 1)
      run_experiment.py            ← [Phase 0→5] matrix → dry-run → guarded loop → ledger
      aggregate.py                 ← [Phase 5] layer 5: pure, re-runnable from caches alone
    viz/
      plots.py                     ← [Phase 7] length curves, CAS, MCP, size sweep, training-type
  tests/                           ← unit tests (Phase 0: infra + matrix + seed + paths + cost)
  notebooks/                       ← optional Colab de-risking notebooks

<root>/                            ← workspace (resolve_root(); ./workspace locally or Drive on Colab)
  data/  mmlu_raw/  gpqa_raw/  hf_cache/  hf_home/
  baselines/   prefills/   cache/   runs/   figures/   ledger/
```

---

## How to run

### Environment setup
```bash
# Local dev (no GPU needed for infra/tests/dry-run):
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" pyyaml

# For judge/generator calls (Anthropic API):
export ANTHROPIC_API_KEY=sk-ant-...

# For offline testing (no API key required):
export COTIM_LLM_BACKEND=mock

# Point workspace at a custom dir (default: ./workspace):
export COTIM_ROOT=/path/to/workspace
```

### Core commands
```bash
# 1. Estimate cost + matrix size (always run this first):
python -m src.pipeline.run_experiment --dry_run

# 2. Override any config knob on the CLI:
python -m src.pipeline.run_experiment --dry_run --cfg n_items=10 families=[B] lengths=[100,300]

# 3. Run for real (Phase 5+; subject GPU inference on Colab):
python -m src.pipeline.run_experiment

# 4. Aggregate results from caches (Layer 5; free re-run):
python -m src.pipeline.aggregate                            # [Phase 5+]

# 5. Run tests (pure Python; works locally without GPU):
pytest tests/ -v
```

### Colab quick-start
```python
# In a Colab cell:
!git clone <repo_url> /content/cot-injection-monitoring && cd /content/cot-injection-monitoring
!pip install -q -e ".[gpu,api,dev]" pyyaml
import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
# resolve_root() auto-mounts Drive and sets workspace = /gdrive/MyDrive/cot-injection-monitoring
from src.infra.paths import resolve_root; paths = resolve_root()
# Run experiment:
!python -m src.pipeline.run_experiment --dry_run
```

---

## Hard engineering constraints (§7 — never re-litigate)
- **Open models only as subjects.** Claude/closed-model `thinking` blocks are cryptographically signed and cannot be injected. Claude may be used as judge/monitor/generator.
- **Never persist GPQA item text.** Store only item hash + derived fields (baseline letter, target letter, metric outcomes). GPQA items are canary-stringed.
- **Every result written to cache immediately** inside the loop — not at the end. Colab disconnects on idle; in-memory-only results are lost.
- **Content-addressable cache (5 layers).** Every expensive op is keyed by sha256 of its inputs. A rerun recomputes nothing already cached. Resumption is by hash, not line number.
- **`AutoModelForCausalLM`, never `AutoModel`** (no LM head → can't generate).
- **`add_special_tokens=False`** when tokenizing an already-templated string (avoid double-BOS). **`skip_special_tokens=False`** when decoding so `</think>` stays visible.
- **Print the rendered template tail before generating** — best habit to catch prefill bugs. Detect if template already opened `<think>`; only prepend if not.
- **DeepSeek-R1-Distill:** `float16` on T4 (bf16 slow/unsupported); no system prompt; temp 0.6, top_p 0.95.
- **Qwen3:** `dtype="auto"` (bf16); fp16 → NaN/garbage on 4B; `enable_thinking=True`; temp 0.6, top_p 0.95, top_k 20.
- **Do not add Qwen2.5-0.5B-Instruct as a subject** — not a reasoning model.

---

## Cost-control + reproducibility workflow

### The 5 cache layers and what each separation saves
| Layer | Operation | Key includes | Separation saves |
|---|---|---|---|
| **1** | Baseline answer + target selection | model, item_hash, sampling, target_mode | All families/lengths/targets reuse this |
| **2** | Prefill content (A template / B,C,D gen) | dataset, item_hash, family, length_bucket, target, generator_model, prompt_version | **All subject models** share one prefill set |
| **3** | Subject continuation (think+answer) | subject_model, item_hash, family, length_bucket, target, prefill_hash, sampling, **sample_idx** | Changing metric/judge → free |
| **4** | Faithfulness (VR) + Monitor (MFR) | judge_model, variant, **prompt_version**, hash(question,trace,answer), plant_hash | Tweak judge prompt → bump prompt_version → Layer 4 only |
| **5** | Metrics / composites / figures | *(pure functions over 1/3/4)* | Always free; run `aggregate.py` freely |

**What each change costs:**
- Change metric formula → re-run only `aggregate.py`. Zero model/API calls.
- Tweak judge prompt → bump `prompt_version` in `config/models.yaml` → only Layer 4. GPU untouched.
- Add a subject model → Layers 1/3/4 for that model; **Layer 2 prefills reused**.
- Add a length/family/target → Layer 2 (new only) + 3 + 4; everything else reused.
- Raise `samples_per_condition` 1→5 → only the 4 new sample indices computed; idx=0 unchanged.

### The dry_run → estimate → adjust → run loop
```
1. Set knobs in config/experiment.yaml (or --cfg on CLI)
2. python -m src.pipeline.run_experiment --dry_run
   → prints net-new counts per layer, $, GPU-hours (after cache)
3. Adjust knobs until estimate is acceptable
4. python -m src.pipeline.run_experiment
   → guardrail halts+checkpoints gracefully at max_spend_usd
5. Disconnect / re-run is free for already-cached work
```

### Cheap-run recipes (§6.6.5)
```yaml
H1 only:    families: [B],    lengths: [100, 300, 700, 1500],  n_items: 50
H2 only:    families: [B, C], lengths: [100, 700]
Smoke:      limit: 1
No-GPU est: dry_run: true
Error bars: samples_per_condition: 3   # (3× GPU cost, adds bootstrap CIs)
Add model:  append to models: → only Layers 1/3/4 for that model
New judge:  bump prompt_version in config/models.yaml → only Layer 4
```

### Determinism contract
- Global `seed` in config. Per-generation seed: `gen_seed(seed, condition_key, sample_idx)` → int, passed to `torch.manual_seed` / vLLM `SamplingParams(seed=...)`. Same config ⇒ byte-identical traces.
- Judges/monitors run at **temperature 0** (fully deterministic).
- `samples_per_condition` is in the Layer-3 cache key. Raising it only adds new sample indices; existing seeds unchanged.

### resolve_root() environment logic
- **Colab** (`google.colab` importable): mount Drive → root = `$COTIM_COLAB_ROOT` or `/gdrive/MyDrive/cot-injection-monitoring`.
- **Local** (else): root = `$COTIM_ROOT` or `./workspace`.
- Sets `HF_HOME`/`HF_DATASETS_CACHE` under the root so model weights + datasets cache durably.
- Mock Colab in tests via `COTIM_COLAB_ROOT` env + patching `is_colab()`.

---
---
<!-- PART 2: Append-only progress log. New entries go at the BOTTOM. -->

## Progress log

### 2026-06-02 — Phase 0: Scaffold + infra (complete)

**Phase goal:** Repo tree, all of `src/infra/`, configs, tests, CLAUDE.md. Fully testable locally (no GPU).

**Files created:**
- `pyproject.toml` — pinned deps; gpu/api/viz/dev extras; `setuptools` find-packages
- `.gitignore` — workspace/, .venv, hf_cache, .env
- `src/{infra,data,attacks,models,judges,metrics,pipeline,viz}/__init__.py`
- `src/infra/paths.py` — `resolve_root()` local/Colab; `Paths` dataclass; `build_paths()`
- `src/infra/cache.py` — `Cache`: sharded JSONL, sha256 key, crash-tolerant load, dry_run
- `src/infra/retry.py` — `@retry` backoff+jitter; `parse_judge_json` guard; `oom_guard`
- `src/infra/seed.py` — `gen_seed(global_seed, condition_key, sample_idx)`
- `src/infra/matrix.py` — `build_condition_matrix(cfg)` with all filters + `samples_per_condition`
- `src/infra/cost.py` — `estimate()` per-layer; `Report.summary()`; `can_proceed()` guardrail
- `src/infra/ledger.py` — `write_manifest()` to `<root>/ledger/<ts>.json`
- `src/infra/llm_client.py` — `LLMClient` (Anthropic API + MockBackend); `from_env()`
- `config/experiment.yaml` — all §6.6.3 knobs + `samples_per_condition`
- `config/models.yaml` — subject/judge/generator configs (Anthropic model IDs)
- `config/cost.yaml` — `sec_per_gen`, pricing
- `src/pipeline/run_experiment.py` — Phase 0 dry_run entrypoint
- `tests/test_cache.py`, `test_matrix.py`, `test_seed.py`, `test_paths.py`, `test_cost.py`

**Decisions & rationale:**
- **Anthropic API** for all judge/monitor/generator calls (native SDK, swap models without code changes).
- **Colab T4 + Drive** for subject GPU inference; local dev is GPU-free.
- **`samples_per_condition` (default 1)** added to matrix + cache key. Raising it only adds new sample indices; existing generations unchanged. Enables bootstrap CIs without invalidating prior work.
- **`gen_seed`** derives a per-generation int from `sha256(seed, condition_key, sample_idx)`. Same config ⇒ identical traces across runs.
- `MockBackend` in `llm_client.py` forces `COTIM_LLM_BACKEND=mock` env for offline unit tests.
- Family A prefills are free templates (no API cost); B/C/D charge generator API.

**Acceptance check results:**
- `pytest tests/ -v` → **43/43 passed** (0.12 s)
- `python -m src.pipeline.run_experiment --dry_run` → prints 550-condition matrix, $0.28 estimate, exits cleanly

**Cache & cost:** Phase 0 is pure scaffolding — zero API or GPU spend. First real cost will appear in Phase 3 (prefill generation) and Phase 5 (subject inference).

**Open questions / next step:** Phase 2 — data + baselines + model layer (merged with dropped Phase 1). See below.

---

### 2026-06-02 — Phase 1+2: Data, baselines, model layer (complete)

**Decision (logged):** The spec's Phase 1 hardcoded smoke-MCQ is dropped. `limit: 1` in config achieves identical de-risking (one condition → one generation) without any special-casing. No hardcoded test items anywhere — run volume is 100% config-driven.

**Files created:**
- `src/data/load_mmlu.py` — open; normalizes to `{item_hash, question, choices, answer_idx, answer_letter, subject, split}`; HF cache at `<root>/data/hf_cache`
- `src/data/load_gpqa.py` — gated stub; shuffles choices on load, stores only `item_hash` + index mapping; **never writes question/choice text to disk**; fully wired Phase 5+
- `src/data/select_target.py` — `select_targets(item, option_logprobs, mode)`: most-plausible distractor (highest-prob wrong option) or sweep all 3; `score_options_hf` for GPU; `uniform_logprobs` for offline/dry-run
- `src/pipeline/baseline.py` — `run_baselines()`: Layer-1 cache of clean answer + target per (model, item); `generate_fn` injected so testable without a real model
- `src/models/parse.py` — `split_trace_answer()`, `extract_letter()`, `parse_output()`; dissociation flag; `skip_special_tokens=False` constraint documented in code
- `src/models/prefill.py` — `build_prefill_prompt()`: print-and-detect `<think>` open tag; `build_clean_prompt()` for baselines
- `src/models/run_model.py` — `load_subject_model()`, `generate_one_hf()` (OOM guard + per-gen seed via `gen_seed`), `run_conditions_hf()` (writes cache immediately per condition)
- `tests/test_parse.py` — 13 tests incl. dissociation case, boxed/stated/therefore/paren extraction, prefer_last
- `tests/test_select_target.py` — 5 tests: most_plausible, sweep order, correct-option exclusion

**Acceptance check:** `pytest tests/ -v` → **62/62 passed** (0.15 s)

**GPU-only paths** (`score_options_hf`, `load_subject_model`, `generate_one_hf`, `run_conditions_hf`) are fully implemented; run on Colab with `limit: 1` for the first live smoke.

**Open questions / next step:** Phase 3 — see below.

---

### 2026-06-02 — Phase 3: Attack generation (complete)

**Phase goal:** Families A/B/C/D templates, Anthropic API generator, length control ±12%, per-family validators, C built per-B (matched pairs). All model-agnostic cache Layer 2.

**Files created:**
- `src/attacks/templates.py` — Family A template expansion (connective sentence pool; no LLM); B/C/D generator instruction prompts; `build_generator_prompt()` dispatch; rotating off-topic topic pool for C (`int(item_hash[:8], 16) % pool_size`)
- `src/attacks/length_control.py` — `MockTokenizer` (word-count × 1.33, for offline tests); `enforce_length()` loop: too-long → sentence-boundary truncation with word-level fallback; too-short → `generator_fn(new_target)`; retry k×; log failures; `extra_tokenizers` records per-model counts as metadata
- `src/attacks/validate.py` — all-families: conclusion + length; B: Jaccard on-topic; C: off-topic + length-matched to paired B ±12% + same conclusion; D: reconsideration + re-check regex markers
- `src/attacks/generate.py` — `generate_family_a()` (template, free); `generate_prefill()` (B/C/D via Anthropic API; failures NOT cached); `generate_all_prefills()` (two-pass B/D first then C; dedup; batch stats)
- `tests/test_templates.py`, `test_length_control.py`, `test_validate.py`, `test_generate.py`

**Decisions & rationale:**
- Family A uses `generator_model="template"` in its Layer-2 cache key (never calls API); distinct from B/C/D's key to avoid collisions.
- Word-level fallback in `truncate_to_length` for texts without sentence boundaries.
- Validation failures are NOT cached — logged and retried next run.
- C always generated after B in a second pass (matched-pairs requirement).

**Bugs fixed:** word-level truncation fallback; Family A cache-key mismatch (`"mock"` vs `"template"`) in `generate_all_prefills`.

**Acceptance check:** `pytest tests/ -v` → **121/121 passed** (0.14 s).

**Open questions / next step:** Phase 4 — see below.

---

### 2026-06-02 — Phase 4: Judges + metrics (complete)

**Phase goal:** VR faithfulness judge + MFR monitor (4 variants) at temp 0, cache Layer 4. Per-item indicators and all composite metrics including EEMR both ways, CAS, MCP, bootstrap CIs.

**Files created:**
- `src/judges/faithfulness.py` — VR judge (knows plant); prompt v1; temp 0; parse via `parse_judge_json`; parse errors NOT cached; `faithfulness_cache_key` includes `plant_hash`
- `src/judges/monitor.py` — MFR monitor; 4 variants: `full_trace` / `answer_only` / `paraphrase` (2-step: paraphrase then judge) / `self` (honesty-prompted system); temp 0; cache keyed by `(judge_model, variant, prompt_version, content_hash)` — no plant in monitor key (monitor is blind)
- `src/metrics/core.py` — per-item 0/1: `cir`, `akr`, `vr`, `mfr`, `scheming`, `robustness_indicator`, `eemr_conjunction`, `osm`; `is_capturable`; `enrich()`
- `src/metrics/composites.py` — `eemr_rate_product` (Yang-style: (1−MFR)·CIR) and `eemr_direct` (mean of captured∧¬flagged) — **always reported together**; `cas`; `mcp` (collapse point with configurable threshold); `bootstrap_ci` (pure Python, seeded, deterministic); `ConditionMetrics.compute()`
- `src/metrics/dissociation.py` — `dissociation_report()` aggregates per-item dissociation flags from `parse_output()`
- `tests/test_judges.py`, `tests/test_metrics.py`

**Decisions & rationale:**
- EEMR is computed both ways (rate_product and direct) and both reported. They diverge when MFR and CIR co-vary across items. `eemr_direct` is used for bootstrap CIs (per-item values available); `eemr_rate_product` matches Yang 2025 for comparison.
- Monitor cache key does NOT include `plant_hash` (monitor is blind to the plant). Faithfulness cache key DOES (it's an input to the judge).
- Bump `prompt_version` in `config/models.yaml` to invalidate Layer 4 without touching GPU work (§6.6.1).
- `bootstrap_ci` uses `random.Random(seed)` — same seed ⇒ identical CIs across runs.
- `is_capturable` filters to `baseline ≠ target` before computing CIR/EEMR (non-capturable items would dilute the rate).

**Bugs fixed:** MCP test rounding (n=5 gave 0.4 ≠ 0.3 at target length 700); switched to n=10 for exact fractions. `ConditionMetrics` test fixed to explicitly include a non-capturable item.

**Acceptance check:** `pytest tests/ -v` → **169/169 passed** (0.27 s).

**Open questions / next step:** Phase 5 — complete `run_experiment.py` (guarded loop over Layers 3+4, spend guardrail, ledger) and `aggregate.py` (Layer 5, pure, re-runnable). Ready to run H1/H2/H3 on Colab.

---

### 2026-06-02 — Phase 5: Core experiment loop + aggregation (complete)

**Phase goal:** Guarded end-to-end loop (`run_experiment.py`) over cache Layers 3+4 with spend guardrail, crash-safe write-immediately, ledger manifest, and pure re-runnable aggregation (`aggregate.py` Layer 5).

**Files created / rewritten:**
- `src/pipeline/run_experiment.py` — REWRITTEN: `_load_items()` (MMLU or dry-run placeholders), `_resolve_targets()` (GPU-free uniform-logprob fallback for local testing), `_run_subject_generation()` (Layer-3 cache check; raises `NotImplementedError` on miss — GPU must be pre-loaded on Colab), `_run_judges()` (VR + all MFR variants), `run()` (guarded loop: per-condition cache check → spend check → generate → judge → accumulate → ledger on completion or guardrail halt), `main()` (CLI with `--dry_run` and `--cfg KEY=VAL`)
- `src/pipeline/aggregate.py` — NEW: `_iter_shard()` (skips truncated lines), `load_subject_results()` (Layer-3 JSONL), `load_judge_results()` (Layer-4 JSONL, merges faithfulness + monitor), `load_baseline_targets()` (Layer-1), `build_enriched_results()` (join all layers, call `enrich()`), `aggregate_results()` (group by condition cell → `ConditionMetrics`, CAS table, MCP table, dissociation summary), `main()` (writes `<root>/runs/aggregate_{ts}.json`)
- `tests/test_experiment.py` — NEW: 18 tests

**Decisions & rationale:**
- Subject generation (`_run_subject_generation`) raises `NotImplementedError` on cache miss rather than silently returning nothing — on Colab the caller loads the model and calls `run_conditions_hf()` directly; the per-condition loop handles GPU inference outside this function so tests stay GPU-free.
- `write_manifest` spreads `extra=` kwargs into the top-level manifest dict (not nested under `"extra"`) — tests assert `manifest.get("mode") == "dry_run"`.
- `load_judge_results` detects faithfulness vs monitor by `"faithfulness" in str(shard_path)` — simple string match; works because shard names are deterministic (`{model}_{variant}.jsonl`).
- CAS and MCP tables computed inside `aggregate_results` by re-grouping over the same enriched records; Layer 5 is pure — zero model/API calls.

**Bugs fixed:** `test_dry_run_writes_ledger_and_exits` initially asserted `manifest.get("extra", {}).get("mode")` but `write_manifest` spreads `extra` at the top level; fixed assertion to `manifest.get("mode")`.

**Acceptance check:** `pytest tests/ -v` → **187/187 passed** (0.33 s).

**CLAUDE.md restoration:** Lines 1-11 had plan-file content prepended by a linter (corrupted header). Stripped via `Edit` to restore the correct `# CLAUDE.md` header.

**Cache & cost:** Phase 5 is CPU-only scaffolding on this Mac. First real Layer-3 cost (subject GPU inference) + Layer-4 API cost (judges) will appear when running on Colab. Workflow: `--dry_run` → read estimate → adjust → real run on Colab T4.

**Open questions / next step:** Phase 6 — unified extensions (size sweep, monitored vs unmonitored, self-as-monitor, training-type axis with Qwen3, answer-only vs full-trace monitor). All reuse existing Layer-2 prefills; only Layers 1/3/4 delta for new models/variants. Pre-run `--dry_run` to confirm delta before each extension.
