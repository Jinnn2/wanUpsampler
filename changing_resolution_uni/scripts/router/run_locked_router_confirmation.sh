#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k_strict}"
SELECTION_JSON="${SELECTION_JSON:-${PROJECT_ROOT}/outputs/router_selection_500_quality_valid_lambda008/selection/architecture_selection.json}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_confirmation_1k_strict_lambda008}"
TRAIN_SEED="${TRAIN_SEED:-42}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cuda}"
OVERHEAD_WARMUP="${OVERHEAD_WARMUP:-20}"
OVERHEAD_REPEATS="${OVERHEAD_REPEATS:-200}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2027}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ ! -f "${SELECTION_JSON}" ]]; then
  echo "Missing validation selection manifest: ${SELECTION_JSON}" >&2
  exit 2
fi

readarray -t selection_values < <(
  python - "${SELECTION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("test_accessed") is not False:
    raise SystemExit("Selection manifest is not validation-only")
print(payload["selected_model_type"])
print(payload["primary_lambda"])
print(payload["split_seed"])
PY
)
SELECTED_MODEL_TYPE="${selection_values[0]}"
PRIMARY_LAMBDA="${selection_values[1]}"
SPLIT_SEED="${selection_values[2]}"

python "${SCRIPT_DIR}/train_router.py" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}" \
  --model_type "${SELECTED_MODEL_TYPE}" \
  --evaluation_stage confirmation \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --primary_lambda "${PRIMARY_LAMBDA}" \
  --split_seed "${SPLIT_SEED}" \
  --seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --require_measured_latency \
  --measure_router_overhead \
  --overhead_warmup "${OVERHEAD_WARMUP}" \
  --overhead_repeats "${OVERHEAD_REPEATS}"

python "${SCRIPT_DIR}/bootstrap_confirmation_test.py" \
  --run-dir "${OUT_DIR}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

if [[ "${SELECTED_MODEL_TYPE}" == "mlp_distill" ]]; then
  python "${SCRIPT_DIR}/analyze_token_attribution.py" \
    --checkpoint "${OUT_DIR}/mlp_distill_router.pt" \
    --dataset_dir "${DATASET_DIR}" \
    --out_dir "${OUT_DIR}/token_attribution_b4" \
    --device "${DEVICE}" \
    --top_k 30
else
  echo "Selected model is ${SELECTED_MODEL_TYPE}; B4 attribution is intentionally skipped." >&2
fi

echo "Locked confirmation complete: ${OUT_DIR}"
