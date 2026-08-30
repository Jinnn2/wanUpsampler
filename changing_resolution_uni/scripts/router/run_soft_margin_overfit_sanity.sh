#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_soft_margin_overfit32_lambda008}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing prepared state dataset: ${DATASET_DIR}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python "${SCRIPT_DIR}/train_soft_margin_router.py" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --model-type soft_margin_pair \
  --train-lambdas 0.08 \
  --eval-lambdas 0.08 \
  --primary-lambda 0.08 \
  --selection-split train \
  --max-train-trajectories 32 \
  --epochs "${EPOCHS}" \
  --batch-size 32 \
  --dropout 0 \
  --weight-decay 0 \
  --device "${DEVICE}" \
  --expected-latency-profile-sha256 "${EXPECTED_LATENCY_PROFILE_SHA256:-}"

echo "Overfit sanity output: ${OUT_DIR}"
echo "This run used 32 existing train trajectories at lambda=0.08 only."
echo "No videos or latent archives were generated or modified."
