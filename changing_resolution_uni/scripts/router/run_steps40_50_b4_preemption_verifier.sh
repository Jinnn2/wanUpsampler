#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
B4_RUNS_ROOT="${B4_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_steps40_50_overall_v1/b4_residual}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/router_steps40_50_b4_preemption_verifier_v1}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024 27182 31415}"
TRAIN_LAMBDAS="${TRAIN_LAMBDAS:-0.01 0.02 0.04 0.06 0.08 0.10}"
EVAL_LAMBDAS="${EVAL_LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"
RISK_THRESHOLDS="${RISK_THRESHOLDS:-0.5 1.0 1.5 2.0}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
MARGIN_TEMPERATURE="${MARGIN_TEMPERATURE:-0.001}"
HARM_EPSILON="${HARM_EPSILON:-0.001}"
CHECKPOINT_RISK_THRESHOLD="${CHECKPOINT_RISK_THRESHOLD:-1.0}"
MAX_HARM_RATE="${MAX_HARM_RATE:-0.02}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DROPOUT="${DROPOUT:-0.1}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4096}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2037}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
read -r -a seed_array <<< "${TRAIN_SEEDS}"
read -r -a train_lambda_array <<< "${TRAIN_LAMBDAS}"
read -r -a eval_lambda_array <<< "${EVAL_LAMBDAS}"
read -r -a threshold_array <<< "${RISK_THRESHOLDS}"

if (( ${#seed_array[@]} != 5 )); then
  echo "TRAIN_SEEDS must contain exactly five verifier initialization seeds." >&2
  exit 2
fi
if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing prepared state dataset: ${DATASET_DIR}" >&2
  echo "Prepared datasets under ${GENERATION_ROOT}:" >&2
  find "${GENERATION_ROOT}" -mindepth 2 -maxdepth 2 -name dataset_manifest.json -printf '  %h\n' >&2
  exit 2
fi
if [[ ! -d "${B4_RUNS_ROOT}" ]]; then
  echo "Missing frozen steps40-50 B4 suite: ${B4_RUNS_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUT_ROOT}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUT_ROOT}" >&2
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

python -m unittest \
  changing_resolution_uni.scripts.router.tests.test_b4_preemption_verifier \
  changing_resolution_uni.scripts.router.tests.test_b4_preemption_summary

for train_seed in "${seed_array[@]}"; do
  seed_out="${OUT_ROOT}/seed_${train_seed}"
  python "${SCRIPT_DIR}/train_b4_preemption_verifier.py" \
    --dataset-dir "${DATASET_DIR}" \
    --b4-runs-root "${B4_RUNS_ROOT}" \
    --out-dir "${seed_out}" \
    --candidate-steps 40 41 42 43 44 45 46 47 48 49 50 \
    --train-lambdas "${train_lambda_array[@]}" \
    --eval-lambdas "${eval_lambda_array[@]}" \
    --primary-lambda "${PRIMARY_LAMBDA}" \
    --radius 3 \
    --margin-temperature "${MARGIN_TEMPERATURE}" \
    --harm-epsilon "${HARM_EPSILON}" \
    --risk-thresholds "${threshold_array[@]}" \
    --checkpoint-risk-threshold "${CHECKPOINT_RISK_THRESHOLD}" \
    --max-validation-harm-rate "${MAX_HARM_RATE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" \
    --seed "${train_seed}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
    --expected-latency-profile-sha256 "${LATENCY_PROFILE_SHA256}"
done

python "${SCRIPT_DIR}/summarize_b4_preemption_verifier.py" \
  --runs-root "${OUT_ROOT}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --max-harm-rate "${MAX_HARM_RATE}" \
  --minimum-positive-train-seeds 4 \
  --minimum-nonnegative-lambdas 8 \
  --minimum-worst-lambda-gain -0.0002

echo "B4-3 sparse preemption selection: ${OUT_ROOT}/selection"
echo "No videos, latent archives, or test records were generated or accessed."
