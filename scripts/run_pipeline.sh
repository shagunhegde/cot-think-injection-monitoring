#!/usr/bin/env bash
# Analysis phase (judges + metrics + plots). Runs all enabled analysis stages in order.
#
#   bash scripts/run_pipeline.sh <config.yaml> [--dry-run] [--cfg KEY=VAL ...]
#
# Stages run in order: faithfulness → monitor → clean_fpr → metrics → plots.
# Each stage is skipped if its `enabled: false` in the config (unless you invoke
# it directly via run_datagen.sh or run_stage.sh).
#
# Logs stream to results/<run_name>/logs/pipeline.log (and the console).
# Set COTIM_PYTHON="uv run python" to use uv instead of the active python.
set -euo pipefail

CFG="${1:?usage: run_pipeline.sh <config.yaml> [--dry-run] [--cfg KEY=VAL ...]}"
shift

PY="${COTIM_PYTHON:-python}"
read_cfg() { $PY -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])) or {}; print((c.get('run',{}) or {}).get(sys.argv[2], sys.argv[3]))" "$CFG" "$1" "$2"; }
RUN_NAME="$(read_cfg run_name run)"
RESULTS_DIR="$(read_cfg results_dir results)"

LOG_DIR="${RESULTS_DIR}/${RUN_NAME}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/pipeline.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "=== pipeline | run=${RUN_NAME} | $(date -u +%FT%TZ) ==="
echo "config: ${CFG} | extra args: $*"
$PY -m src.pipeline.stages --config "${CFG}" --stage all_analysis "$@"
echo "=== pipeline complete | $(date -u +%FT%TZ) ==="
