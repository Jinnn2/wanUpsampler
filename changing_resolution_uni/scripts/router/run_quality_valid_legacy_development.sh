#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_quality_valid}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${PROJECT_ROOT}/outputs/oracle_500_quality_valid_sweep}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/outputs/router_500_quality_valid_lambda_sweep}"
TRAIN_LAMBDAS="${TRAIN_LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"
RUN_TRAINING="${RUN_TRAINING:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MODEL_TYPE="${MODEL_TYPE:-mlp_distill}"

if [[ "${MODEL_TYPE}" != "all" && "${MODEL_TYPE}" != "mlp_distill" ]]; then
  echo "B4 token attribution requires MODEL_TYPE=all or mlp_distill." >&2
  exit 2
fi

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
  read -r -a lambda_array <<< "${TRAIN_LAMBDAS}"
  for lambda_value in "${lambda_array[@]}"; do
    lambda_slug="${lambda_value//./}"
    lambda_out="${TRAIN_ROOT}/lambda_${lambda_slug}"
    if [[ "${SKIP_COMPLETED}" == "1" \
      && -f "${lambda_out}/router_benchmark_results.csv" \
      && -f "${lambda_out}/router_benchmark_summary.json" \
      && -f "${lambda_out}/token_attribution_b4/top_late_switch_words.csv" \
      && -f "${lambda_out}/token_attribution_b4/attribution_metadata.json" ]]; then
      echo "Skipping completed lambda=${lambda_value}: ${lambda_out}"
      continue
    fi
    echo "Training development routers at lambda=${lambda_value}"
    DATASET_DIR="${DATASET_DIR}" \
    OUT_DIR="${lambda_out}" \
    PRIMARY_LAMBDA="${lambda_value}" \
    MODEL_TYPE="${MODEL_TYPE}" \
    ALLOW_ESTIMATED_LATENCY=1 \
    bash "${SCRIPT_DIR}/run_train_and_benchmark.sh"
  done
  python "${SCRIPT_DIR}/summarize_lambda_router_runs.py" \
    --runs-root "${TRAIN_ROOT}"
else
  echo "Training skipped. Set RUN_TRAINING=1 to enable the development run."
fi

echo "Development dataset: ${DATASET_DIR}"
echo "Lambda sweep      : ${ANALYSIS_DIR}"
echo "Router outputs    : ${TRAIN_ROOT}/lambda_*"
