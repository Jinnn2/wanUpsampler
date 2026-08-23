#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_quality_valid}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/token_attribution_input_audit}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/audit_token_attribution_inputs.py" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}"

echo "Send this report back for diagnosis:"
echo "${OUT_DIR}/t5_attribution_input_audit.json"
