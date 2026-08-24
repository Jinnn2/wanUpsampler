#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_quality_valid}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/router_selection_500_quality_valid_lambda008}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024 31415 27182}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2027}"
DEVICE="${DEVICE:-cuda}"
ALLOW_ESTIMATED_LATENCY="${ALLOW_ESTIMATED_LATENCY:-1}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
read -r -a seed_array <<< "${TRAIN_SEEDS}"
if (( ${#seed_array[@]} < 3 )); then
  echo "TRAIN_SEEDS must contain at least three initialization seeds." >&2
  exit 2
fi

for train_seed in "${seed_array[@]}"; do
  seed_out="${OUT_ROOT}/seed_${train_seed}"
  args=(
    --dataset_dir "${DATASET_DIR}"
    --out_dir "${seed_out}"
    --model_type all
    --evaluation_stage selection
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --primary_lambda "${PRIMARY_LAMBDA}"
    --split_seed "${SPLIT_SEED}"
    --seed "${train_seed}"
    --device "${DEVICE}"
  )
  if [[ "${ALLOW_ESTIMATED_LATENCY}" == "1" ]]; then
    args+=(--allow_estimated_latency)
  fi
  python "${SCRIPT_DIR}/train_router.py" "${args[@]}"
done

python "${SCRIPT_DIR}/summarize_multiseed_selection.py" \
  --runs-root "${OUT_ROOT}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

echo "Validation-only architecture selection: ${OUT_ROOT}/selection/architecture_selection.json"
echo "No test split was accessed by this launcher."
