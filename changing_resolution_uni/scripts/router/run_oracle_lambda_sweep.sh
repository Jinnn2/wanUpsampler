#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k_router_ready}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/oracle_lambda_sweep_1k}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/sweep_oracle_lambda.py" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --lambda-min "${LAMBDA_MIN:-0.001}" \
  --lambda-max "${LAMBDA_MAX:-0.100}" \
  --lambda-step "${LAMBDA_STEP:-0.001}" \
  --near-tie-threshold "${NEAR_TIE_THRESHOLD:-0.001}"
