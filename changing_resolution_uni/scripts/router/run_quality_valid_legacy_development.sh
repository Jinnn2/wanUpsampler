#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_quality_valid}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${PROJECT_ROOT}/outputs/oracle_500_quality_valid_sweep}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_500_quality_valid_lambda${PRIMARY_LAMBDA//./}}"
RUN_TRAINING="${RUN_TRAINING:-1}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  if [[ -e "${DATASET_DIR}" ]]; then
    echo "Output exists without a completed manifest: ${DATASET_DIR}" >&2
    exit 2
  fi
  python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/prepare_quality_valid_legacy_dataset.py" \
    --source-dir "${SOURCE_DIR}" \
    --output-dir "${DATASET_DIR}" \
    --primary-lambda "${PRIMARY_LAMBDA}"
else
  echo "Reusing prepared development dataset: ${DATASET_DIR}"
fi

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/cleanup_legacy_records.py" \
  --dataset_dir "${DATASET_DIR}" \
  --profile quality_valid_legacy \
  --strict

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/sweep_oracle_lambda.py" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${ANALYSIS_DIR}" \
  --lambda-min 0.001 \
  --lambda-max 0.100 \
  --lambda-step 0.001 \
  --allow-estimated-latency

if [[ "${RUN_TRAINING}" == "1" ]]; then
  DATASET_DIR="${DATASET_DIR}" \
  OUT_DIR="${OUT_DIR}" \
  PRIMARY_LAMBDA="${PRIMARY_LAMBDA}" \
  ALLOW_ESTIMATED_LATENCY=1 \
  bash "${SCRIPT_DIR}/run_train_and_benchmark.sh"
else
  echo "Training skipped. Set RUN_TRAINING=1 to enable the development run."
fi

echo "Development dataset: ${DATASET_DIR}"
echo "Lambda sweep      : ${ANALYSIS_DIR}"
echo "Router outputs    : ${OUT_DIR}"
