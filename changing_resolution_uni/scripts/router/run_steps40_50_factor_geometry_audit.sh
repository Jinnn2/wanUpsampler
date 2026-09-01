#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_steps40_50_factor_geometry_audit_v2}"
SHUFFLE_REPETITIONS="${SHUFFLE_REPETITIONS:-3}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing state dataset manifest: ${DATASET_DIR}/dataset_manifest.json" >&2
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

python "${SCRIPT_DIR}/analyze_steps40_50_factor_geometry.py" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --candidate-steps 40 41 42 43 44 45 46 47 48 49 50 \
  --shuffle-repetitions "${SHUFFLE_REPETITIONS}"

echo "Factor geometry audit: ${OUT_DIR}/factor_geometry_report.json"
echo "No videos, latents, router checkpoints, or test records were modified."
