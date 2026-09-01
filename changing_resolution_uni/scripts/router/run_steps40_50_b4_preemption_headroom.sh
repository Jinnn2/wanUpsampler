#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
B4_RUNS_ROOT="${B4_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_steps40_50_overall_v1/b4_residual}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_steps40_50_b4_preemption_headroom_v1}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-128}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing state dataset manifest: ${DATASET_DIR}/dataset_manifest.json" >&2
  exit 2
fi
if [[ ! -d "${B4_RUNS_ROOT}" ]]; then
  echo "Missing steps40-50 B4 runs root: ${B4_RUNS_ROOT}" >&2
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

python -m unittest \
  changing_resolution_uni.scripts.router.tests.test_b4_preemption_headroom

python "${SCRIPT_DIR}/audit_steps40_50_b4_preemption_headroom.py" \
  --dataset-dir "${DATASET_DIR}" \
  --b4-runs-root "${B4_RUNS_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --candidate-steps 40 41 42 43 44 45 46 47 48 49 50 \
  --device "${DEVICE}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"

echo "B4 preemption headroom: ${OUT_DIR}/b4_preemption_headroom_report.json"
echo "No videos, latents, router checkpoints, or test records were modified."
