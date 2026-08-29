#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_selection}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024 31415 27182}"
TRAIN_LAMBDAS="${TRAIN_LAMBDAS:-0.01 0.02 0.04 0.06 0.08 0.10}"
EVAL_LAMBDAS="${EVAL_LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
FEATURE_GROUPS="${FEATURE_GROUPS:-x0_global residual_global x0_channel residual_channel local_energy trajectory_delta}"
HARM_EPSILON="${HARM_EPSILON:-0.001}"
RISK_THRESHOLD="${RISK_THRESHOLD:-0.5}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_BATCH_TRAJECTORIES="${EVAL_BATCH_TRAJECTORIES:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2027}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
read -r -a seed_array <<< "${TRAIN_SEEDS}"
read -r -a train_lambda_array <<< "${TRAIN_LAMBDAS}"
read -r -a eval_lambda_array <<< "${EVAL_LAMBDAS}"
read -r -a feature_group_array <<< "${FEATURE_GROUPS}"
if (( ${#seed_array[@]} < 3 )); then
  echo "TRAIN_SEEDS must contain at least three initialization seeds." >&2
  exit 2
fi
if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing prepared variable-lambda state dataset: ${DATASET_DIR}" >&2
  exit 2
fi
LATENCY_PROFILE_SHA256="${EXPECTED_LATENCY_PROFILE_SHA256:-$(python - "${DATASET_DIR}/dataset_manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = str(manifest.get("latency_profile", {}).get("sha256", ""))
if len(value) != 64:
    raise SystemExit("Dataset manifest has no locked latency profile SHA256")
print(value)
PY
)}"
if [[ -e "${OUT_ROOT}/selection/architecture_selection.json" ]]; then
  echo "Selection already exists; refusing to overwrite: ${OUT_ROOT}" >&2
  exit 2
fi

for train_seed in "${seed_array[@]}"; do
  seed_out="${OUT_ROOT}/seed_${train_seed}"
  if [[ -e "${seed_out}" ]]; then
    echo "Seed output already exists; refusing to mix runs: ${seed_out}" >&2
    exit 2
  fi
  python "${SCRIPT_DIR}/train_variable_lambda_router.py" \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${seed_out}" \
    --model-type both \
    --feature-groups "${feature_group_array[@]}" \
    --train-lambdas "${train_lambda_array[@]}" \
    --eval-lambdas "${eval_lambda_array[@]}" \
    --primary-lambda "${PRIMARY_LAMBDA}" \
    --harm-epsilon "${HARM_EPSILON}" \
    --risk-threshold "${RISK_THRESHOLD}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --seed "${train_seed}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --eval-batch-trajectories "${EVAL_BATCH_TRAJECTORIES}" \
    --expected-latency-profile-sha256 "${LATENCY_PROFILE_SHA256}"
done

python "${SCRIPT_DIR}/summarize_variable_lambda_runs.py" \
  --runs-root "${OUT_ROOT}" \
  --reference-model prompt_only \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

echo "Variable-lambda validation selection: ${OUT_ROOT}/selection"
echo "Locked latency profile SHA256: ${LATENCY_PROFILE_SHA256}"
echo "Test prompt states were not prepared or accessed."
