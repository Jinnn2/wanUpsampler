#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_quality_valid}"
TOKEN_T5_DIR="${TOKEN_T5_DIR:-${DATASET_DIR}/token_attribution_embeddings}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/outputs/router_500_quality_valid_lambda_sweep}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
TRAIN_LAMBDAS="${TRAIN_LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rebuild_args=(
  --dataset-dir "${DATASET_DIR}"
  --out-dir "${TOKEN_T5_DIR}"
  --model-root "${MODEL_ROOT}"
)
if [[ -n "${TOKENIZER_PATH}" ]]; then
  rebuild_args+=(--tokenizer-path "${TOKENIZER_PATH}")
fi
python "${SCRIPT_DIR}/rebuild_token_attribution_metadata.py" "${rebuild_args[@]}"

python "${SCRIPT_DIR}/audit_token_attribution_inputs.py" \
  --dataset-dir "${DATASET_DIR}" \
  --t5-dir "${TOKEN_T5_DIR}" \
  --out-dir "${TRAIN_ROOT}/token_input_audit" \
  --strict

read -r -a lambda_array <<< "${TRAIN_LAMBDAS}"
for lambda_value in "${lambda_array[@]}"; do
  lambda_slug="${lambda_value//./}"
  lambda_out="${TRAIN_ROOT}/lambda_${lambda_slug}"
  checkpoint="${lambda_out}/linear_ordinal_router.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing trained checkpoint for lambda=${lambda_value}: ${checkpoint}" >&2
    exit 2
  fi
  echo "Rebuilding natural-word attribution for lambda=${lambda_value}"
  python "${SCRIPT_DIR}/analyze_token_attribution.py" \
    --checkpoint "${checkpoint}" \
    --dataset_dir "${DATASET_DIR}" \
    --t5_dir "${TOKEN_T5_DIR}" \
    --out_dir "${lambda_out}/token_attribution" \
    --top_k 30
  python "${SCRIPT_DIR}/print_results_summary.py" \
    --out_dir "${lambda_out}" \
    --dataset_dir "${DATASET_DIR}"
done

python "${SCRIPT_DIR}/summarize_lambda_router_runs.py" \
  --runs-root "${TRAIN_ROOT}"

echo "Token metadata and all lambda attributions are complete."
echo "Token embedding directory: ${TOKEN_T5_DIR}"
