#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection}"
CONTROL_DATASET_DIR="${CONTROL_DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_train800_control200_v1}"
CONTROL_SPLIT_SALT="train800_control200_v1"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_train800_control200_b4_deterministic_eval_v1}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${SOURCE_DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing source state dataset: ${SOURCE_DATASET_DIR}" >&2
  exit 2
fi

if [[ ! -e "${CONTROL_DATASET_DIR}" ]]; then
  python "${SCRIPT_DIR}/prepare_train800_control_split.py" \
    --source-dataset-dir "${SOURCE_DATASET_DIR}" \
    --output-dir "${CONTROL_DATASET_DIR}" \
    --validation-count 200 \
    --expected-source-prompts 1000 \
    --expected-base-seed 42 \
    --split-salt "${CONTROL_SPLIT_SALT}"
fi

python - "${CONTROL_DATASET_DIR}/dataset_manifest.json" "${CONTROL_SPLIT_SALT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_salt = sys.argv[2]
manifest = json.loads(path.read_text(encoding="utf-8"))
derivation = manifest.get("derivation", {})
assert manifest["schema"] == "variable_lambda_online_state_dataset_v1"
assert manifest["selected_splits"] == ["train", "validation"]
assert manifest["test_accessed"] is False
assert manifest["is_complete"] is True
assert manifest["splits"]["train"]["prompt_count"] == 800
assert manifest["splits"]["train"]["trajectory_count"] == 800
assert manifest["splits"]["validation"]["prompt_count"] == 200
assert manifest["splits"]["validation"]["trajectory_count"] == 200
assert derivation["schema"] == "train800_control200_hash_split_v1"
assert derivation["split_salt"] == expected_salt
assert derivation["source_validation_index_accessed"] is False
assert derivation["source_test_accessed"] is False
print(f"Control split verified: {path}")
PY

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Control split preparation complete; GPU training was not started."
  exit 0
fi

export DATASET_DIR="${CONTROL_DATASET_DIR}"
export OUT_ROOT
export MODEL_TYPE=all

exec bash "${SCRIPT_DIR}/run_multiseed_variable_lambda_selection.sh"
