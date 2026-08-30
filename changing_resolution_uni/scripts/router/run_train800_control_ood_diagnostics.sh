#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_train800_control200_b4_deterministic_eval_v1}"
CONTROL_DATASET_DIR="${CONTROL_DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_train800_control200_v1}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/ood_diagnostics}"
REFERENCE_RUNS_ROOT="${REFERENCE_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_b4_hybrid_deterministic_eval_v1}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_TRAJECTORIES="${EVAL_BATCH_TRAJECTORIES:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2027}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${OOD_DATASET_DIR:-}" ]]; then
  if [[ ! -f "${CONTROL_DATASET_DIR}/dataset_manifest.json" ]]; then
    echo "Missing control dataset manifest: ${CONTROL_DATASET_DIR}/dataset_manifest.json" >&2
    exit 2
  fi
  OOD_DATASET_DIR="$(python - "${CONTROL_DATASET_DIR}/dataset_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source_manifest = Path(manifest["derivation"]["source_dataset_manifest"])
print(source_manifest.parent)
PY
)"
fi
if [[ ! -f "${OOD_DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing OOD source state dataset: ${OOD_DATASET_DIR}" >&2
  exit 2
fi
echo "OOD source state dataset: ${OOD_DATASET_DIR}"

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
