#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/router_fixed_guard_v1}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024 31415 27182}"
EVAL_LAMBDAS="${EVAL_LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2043}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${DATASET_DIR:-}" ]]; then
  candidates=()
  for candidate in "${GENERATION_ROOT}"/router_variable_lambda_states_selection*; do
    [[ -f "${candidate}/dataset_manifest.json" ]] && candidates+=("${candidate}")
  done
  if (( ${#candidates[@]} != 1 )); then
    echo "Expected one complete selection state dataset; found ${#candidates[@]}." >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 2
  fi
  DATASET_DIR="${candidates[0]}"
fi

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing state dataset: ${DATASET_DIR}" >&2
  exit 2
fi
if [[ -e "${OUT_ROOT}/selection/architecture_selection.json" ]]; then
  echo "Selection already exists: ${OUT_ROOT}" >&2
  exit 2
fi

LATENCY_SHA="${EXPECTED_LATENCY_PROFILE_SHA256:-$(python - "${DATASET_DIR}/dataset_manifest.json" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())["latency_profile"]["sha256"]
print(value)
PY
)}"

read -r -a seed_array <<< "${TRAIN_SEEDS}"
read -r -a lambda_array <<< "${EVAL_LAMBDAS}"
for train_seed in "${seed_array[@]}"; do
  seed_out="${OUT_ROOT}/seed_${train_seed}"
  python "${SCRIPT_DIR}/train_fixed_guard_router.py" \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${seed_out}" \
    --eval-lambdas "${lambda_array[@]}" \
    --primary-lambda "${PRIMARY_LAMBDA}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${train_seed}" \
    --device "${DEVICE}" \
    --expected-latency-profile-sha256 "${LATENCY_SHA}"
done

python "${SCRIPT_DIR}/summarize_variable_lambda_runs.py" \
  --runs-root "${OUT_ROOT}" \
  --reference-model fixed_guard_prompt \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

echo "Fixed-Guard selection: ${OUT_ROOT}/selection"
echo "Primary result: ${OUT_ROOT}/selection/paired_fixed_deltas.csv"
echo "State ablation: ${OUT_ROOT}/selection/paired_reference_deltas.csv"
echo "Test split was not accessed."
