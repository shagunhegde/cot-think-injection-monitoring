#!/usr/bin/env bash
# Datagen phase (GPU + attacker API). Runs one stage, or all enabled datagen stages.
#
#   bash scripts/run_datagen.sh <config.yaml> [stage] [--shard i --total-shards N] [--dry-run]
#
# stage ∈ {baselines, prefills, subject, clean_baseline, all_datagen}  (default: all_datagen)
# An explicitly named stage runs regardless of its `enabled:` flag; `all_datagen` honours it.
#
# Logs stream to results/<run_name>/logs/datagen_<stage>.log (and the console).
# Set COTIM_PYTHON="uv run python" to use uv instead of the active python.
set -euo pipefail

CFG="${1:?usage: run_datagen.sh <config.yaml> [stage] [--shard i --total-shards N] [--dry-run]}"
shift
STAGE="all_datagen"
if [[ $# -gt 0 && "$1" != -* ]]; then STAGE="$1"; shift; fi

PY="${COTIM_PYTHON:-python}"
read_cfg() { $PY -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])) or {}; print((c.get('run',{}) or {}).get(sys.argv[2], sys.argv[3]))" "$CFG" "$1" "$2"; }
RUN_NAME="$(read_cfg run_name run)"
RESULTS_DIR="$(read_cfg results_dir results)"

LOG_DIR="${RESULTS_DIR}/${RUN_NAME}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/datagen_${STAGE}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "=== datagen | run=${RUN_NAME} | stage=${STAGE} | $(date -u +%FT%TZ) ==="
echo "config: ${CFG} | extra args: $*"
$PY -m src.pipeline.stages --config "${CFG}" --stage "${STAGE}" "$@"
echo "=== datagen ${STAGE} complete ==="
