#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/oracle_strict_analysis_1k}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/cleanup_legacy_records.py" \
  --dataset_dir "${DATASET_DIR}" \
  --profile formal \
  --strict

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/analyze_oracle_dimensions.py" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}/dimensions"

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/sweep_oracle_lambda.py" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}/lambda_sweep" \
  --lambda-min 0.001 \
  --lambda-max 0.100 \
  --lambda-step 0.001

echo "Strict oracle analysis complete: ${OUT_DIR}"
