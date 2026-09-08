#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_quality_valid}"
B4_RUN_ROOT="${B4_RUN_ROOT:-${PROJECT_ROOT}/outputs/router_selection_500_quality_valid_lambda008}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/continuous_budget_prior_legacy_lambda008}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
TARGET_TYPE="${TARGET_TYPE:-soft_expected}"
SOFT_TARGET_TAU="${SOFT_TARGET_TAU:-0.02}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024 31415 27182}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
HUBER_BETA="${HUBER_BETA:-0.02}"
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
  b4_checkpoint="${B4_RUN_ROOT}/seed_${train_seed}/mlp_distill_router.pt"
  seed_out="${OUT_ROOT}/seed_${train_seed}"
  if [[ ! -f "${b4_checkpoint}" ]]; then
    echo "Missing frozen B4 checkpoint: ${b4_checkpoint}" >&2
    exit 2
  fi
  if [[ -e "${seed_out}/continuous_budget_validation_summary.json" ]]; then
    echo "Seed output already exists; refusing to overwrite: ${seed_out}" >&2
    exit 2
  fi
  args=(
    --dataset-dir "${DATASET_DIR}"
    --b4-checkpoint "${b4_checkpoint}"
    --out-dir "${seed_out}"
    --target-type "${TARGET_TYPE}"
    --primary-lambda "${PRIMARY_LAMBDA}"
    --soft-target-tau "${SOFT_TARGET_TAU}"
    --split-seed "${SPLIT_SEED}"
    --seed "${train_seed}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --huber-beta "${HUBER_BETA}"
    --device "${DEVICE}"
  )
  if [[ "${ALLOW_ESTIMATED_LATENCY}" == "1" ]]; then
    args+=(--allow-estimated-latency)
  else
    args+=(--require-measured-latency)
  fi
  python "${SCRIPT_DIR}/train_continuous_budget_prior.py" "${args[@]}"
done

python "${SCRIPT_DIR}/summarize_continuous_budget_prior.py" \
  --runs-root "${OUT_ROOT}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

echo "Continuous-budget validation summary: ${OUT_ROOT}/selection/continuous_budget_prior_selection.json"
echo "No test split was evaluated by this launcher."
