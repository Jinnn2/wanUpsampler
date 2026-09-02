#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
B4_RUNS_ROOT="${B4_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_soft_margin_v1}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_b4_conditional_state_capacity_v1}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-128}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing state dataset manifest: ${DATASET_DIR}/dataset_manifest.json" >&2
  exit 2
fi
if [[ ! -d "${B4_RUNS_ROOT}" ]]; then
  echo "Missing frozen B4 runs root: ${B4_RUNS_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUT_DIR}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python - <<'PY'
import sklearn
import torch
print(f"torch={torch.__version__} sklearn={sklearn.__version__}")
PY

python "${SCRIPT_DIR}/analyze_b4_conditional_state_capacity.py" \
  --dataset-dir "${DATASET_DIR}" \
  --b4-runs-root "${B4_RUNS_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"

echo "B4 conditional state capacity audit: ${OUT_DIR}/conditional_state_capacity_report.json"
echo "Frozen B4 checkpoints and the locked test split were not modified or accessed."
