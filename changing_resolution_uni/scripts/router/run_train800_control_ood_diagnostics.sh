#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_train800_control200_b4_deterministic_eval_v1}"
OOD_DATASET_DIR="${OOD_DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/ood_diagnostics}"
REFERENCE_RUNS_ROOT="${REFERENCE_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_b4_hybrid_deterministic_eval_v1}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_TRAJECTORIES="${EVAL_BATCH_TRAJECTORIES:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2027}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

args=(
  --runs-root "${RUNS_ROOT}"
  --ood-dataset-dir "${OOD_DATASET_DIR}"
  --out-dir "${OUT_DIR}"
  --base-seeds 42 100 2024
  --device "${DEVICE}"
  --eval-batch-trajectories "${EVAL_BATCH_TRAJECTORIES}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
)
if [[ -d "${REFERENCE_RUNS_ROOT}" ]]; then
  args+=(--reference-runs-root "${REFERENCE_RUNS_ROOT}")
else
  echo "Reference train1000 runs not found; training-size comparison skipped: ${REFERENCE_RUNS_ROOT}" >&2
fi

python "${SCRIPT_DIR}/evaluate_train800_control_ood.py" "${args[@]}"
