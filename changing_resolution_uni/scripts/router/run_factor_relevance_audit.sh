#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
B4_RUNS_ROOT="${B4_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_soft_margin_v1}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_factor_relevance_audit_v1}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SEED_SHUFFLE_REPETITIONS="${SEED_SHUFFLE_REPETITIONS:-20}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-128}"

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
import sklearn
import torch
print(f"torch={torch.__version__} sklearn={sklearn.__version__}")
PY

python "${SCRIPT_DIR}/analyze_factor_relevance.py" \
  --dataset-dir "${DATASET_DIR}" \
  --b4-runs-root "${B4_RUNS_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --seed-shuffle-repetitions "${SEED_SHUFFLE_REPETITIONS}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"

echo "Factor relevance audit: ${OUT_DIR}/factor_relevance_report.json"
echo "No videos, latent archives, router checkpoints, or test records were modified."
