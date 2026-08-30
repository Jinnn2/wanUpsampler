#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_soft_margin_v1}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024 31415 27182}"
TRAIN_LAMBDAS="${TRAIN_LAMBDAS:-0.01 0.02 0.04 0.06 0.08 0.10}"
EVAL_LAMBDAS="${EVAL_LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
FEATURE_GROUPS="${FEATURE_GROUPS:-x0_global residual_global x0_channel residual_channel local_energy trajectory_delta}"
HARM_EPSILON="${HARM_EPSILON:-0.001}"
RISK_MARGIN="${RISK_MARGIN:-0.0}"
MARGIN_TEMPERATURE="${MARGIN_TEMPERATURE:-0.02}"
B4_TEMPERATURE="${B4_TEMPERATURE:-0.02}"
B4_EMD_WEIGHT="${B4_EMD_WEIGHT:-0.5}"
RESIDUAL_LOGIT_LIMIT="${RESIDUAL_LOGIT_LIMIT:-4.0}"
RESIDUAL_PENALTY_WEIGHT="${RESIDUAL_PENALTY_WEIGHT:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DROPOUT="${DROPOUT:-0.1}"
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
  echo "Missing prepared state dataset: ${DATASET_DIR}" >&2
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
  python "${SCRIPT_DIR}/train_soft_margin_router.py" \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${seed_out}" \
    --model-type soft_margin_pair \
    --feature-groups "${feature_group_array[@]}" \
    --train-lambdas "${train_lambda_array[@]}" \
    --eval-lambdas "${eval_lambda_array[@]}" \
    --primary-lambda "${PRIMARY_LAMBDA}" \
    --harm-epsilon "${HARM_EPSILON}" \
    --risk-margin "${RISK_MARGIN}" \
    --margin-temperature "${MARGIN_TEMPERATURE}" \
    --b4-temperature "${B4_TEMPERATURE}" \
    --b4-emd-weight "${B4_EMD_WEIGHT}" \
    --residual-logit-limit "${RESIDUAL_LOGIT_LIMIT}" \
    --residual-penalty-weight "${RESIDUAL_PENALTY_WEIGHT}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" \
    --seed "${train_seed}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --eval-batch-trajectories "${EVAL_BATCH_TRAJECTORIES}" \
    --expected-latency-profile-sha256 "${LATENCY_PROFILE_SHA256}"
done

python "${SCRIPT_DIR}/summarize_variable_lambda_runs.py" \
  --runs-root "${OUT_ROOT}" \
  --reference-model b4_offline \
  --secondary-reference-model soft_margin_control \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

echo "Soft-margin validation selection: ${OUT_ROOT}/selection"
echo "No videos or latent archives were generated or modified."
echo "Test prompt states were not prepared or accessed."
