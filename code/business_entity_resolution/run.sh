#!/usr/bin/env bash
# One-command end-to-end run on Linux (cluster node).
#   bash run.sh                      # full run
#   bash run.sh --from stage2_fit    # resume from a step
# Optional environment overrides (defaults shown in CLUSTER_RUN.md):
#   ER_DATA_DIR  ER_WORK_DIR  ER_OUTPUT_DIR  ER_N_JOBS  ER_BACKEND=xgb|lgb  ER_VALIDATOR
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
mkdir -p "${ER_WORK_DIR:-../../work}"
python src/run_all.py --check
python src/run_all.py "$@" 2>&1 | tee -a "${ER_WORK_DIR:-../../work}/run_all.log"
