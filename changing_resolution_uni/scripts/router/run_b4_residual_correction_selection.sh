#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
B4_RUNS_ROOT="${B4_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_soft_margin_v1}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_b4_residual_correction_selection_v1}"
DEVICE="${DEVICE:-cuda}"
CV_FOLDS="${CV_FOLDS:-3}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-128}"
RIDGE_ALPHAS="${RIDGE_ALPHAS:-1 10 100 1000}"
CORRECTION_SCALES="${CORRECTION_SCALES:-0.05 0.10 0.20 0.40}"
GATE_THRESHOLDS="${GATE_THRESHOLDS:-0.25 0.50 1.0 2.0}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing state dataset manifest: ${DATASET_DIR}/dataset_manifest.json" >&2
  exit 2
fi
if [[ ! -d "${B4_RUNS_ROOT}" ]]; then
  echo "Missing B4 runs root: ${B4_RUNS_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUT_DIR}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python - <<'PY'
import joblib
import sklearn
import torch

print(
    f"torch={torch.__version__} sklearn={sklearn.__version__} "
    f"joblib={joblib.__version__}"
)
PY

python -m unittest \
  changing_resolution_uni.scripts.router.tests.test_b4_residual_correction

# Grid strings intentionally undergo word splitting so each value becomes one CLI item.
# shellcheck disable=SC2086
python "${SCRIPT_DIR}/select_b4_residual_correction.py" \
  --dataset-dir "${DATASET_DIR}" \
  --b4-runs-root "${B4_RUNS_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --cv-folds "${CV_FOLDS}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --ridge-alphas ${RIDGE_ALPHAS} \
  --correction-scales ${CORRECTION_SCALES} \
  --gate-thresholds ${GATE_THRESHOLDS}

echo "Residual correction report: ${OUT_DIR}/residual_correction_report.json"
echo "This run reads train and validation only; test data is not accessed."
echo "No videos or latent archives were generated or modified."
