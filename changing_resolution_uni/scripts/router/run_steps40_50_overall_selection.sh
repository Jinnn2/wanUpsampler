#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
OVERALL_ROOT="${OVERALL_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_steps40_50_overall_v1}"
RESIDUAL_ROOT="${OVERALL_ROOT}/b4_residual"
SOFT_ROOT="${OVERALL_ROOT}/soft_margin"
OVERALL_SELECTION_DIR="${OVERALL_ROOT}/overall_selection"

export DATASET_DIR
export CANDIDATE_STEPS="40 41 42 43 44 45 46 47 48 49 50"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024 31415 27182}"
export TRAIN_LAMBDAS="${TRAIN_LAMBDAS:-0.01 0.02 0.04 0.06 0.08 0.10}"
export EVAL_LAMBDAS="${EVAL_LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"
export PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
export FEATURE_GROUPS="${FEATURE_GROUPS:-x0_global residual_global x0_channel residual_channel local_energy trajectory_delta}"
export HARM_EPSILON="${HARM_EPSILON:-0.001}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-0.0003}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
export DROPOUT="${DROPOUT:-0.1}"
export B4_TEMPERATURE="${B4_TEMPERATURE:-0.02}"
export B4_EMD_WEIGHT="${B4_EMD_WEIGHT:-0.5}"
export DEVICE="${DEVICE:-cuda}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export EVAL_BATCH_TRAJECTORIES="${EVAL_BATCH_TRAJECTORIES:-64}"
export BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2029}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing state dataset manifest: ${DATASET_DIR}/dataset_manifest.json" >&2
  exit 2
fi
if [[ -e "${OVERALL_ROOT}" ]]; then
  echo "Overall output already exists; refusing to mix runs: ${OVERALL_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m unittest \
  changing_resolution_uni.scripts.router.tests.test_candidate_step_subset \
  changing_resolution_uni.scripts.router.tests.test_steps40_50_overall_summary

echo "[1/3] Training B4 anchored residual suite on steps 40-50..."
OUT_ROOT="${RESIDUAL_ROOT}" MODEL_TYPE=b4_residual_pair \
  bash "${SCRIPT_DIR}/run_multiseed_variable_lambda_selection.sh"

echo "[2/3] Training causal soft-margin suite on steps 40-50..."
OUT_ROOT="${SOFT_ROOT}" B4_CHECKPOINT_ROOT="${RESIDUAL_ROOT}" \
  bash "${SCRIPT_DIR}/run_multiseed_variable_lambda_soft_margin_selection.sh"

echo "[3/3] Verifying matched B4 predictions and writing overall comparison..."
python "${SCRIPT_DIR}/summarize_steps40_50_overall.py" \
  --residual-runs-root "${RESIDUAL_ROOT}" \
  --soft-runs-root "${SOFT_ROOT}" \
  --out-dir "${OVERALL_SELECTION_DIR}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

echo "Overall 40-50 selection: ${OVERALL_SELECTION_DIR}/overall_selection.json"
echo "No videos or latent archives were generated or modified."
echo "Only train and validation state records were accessed; test remains untouched."
