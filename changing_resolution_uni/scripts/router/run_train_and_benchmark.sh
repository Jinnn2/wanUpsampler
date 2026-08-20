#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_benchmarks_1k}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.05}"
SEED="${SEED:-42}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "================================================================================"
echo " [Step 1/3] Merging & Verifying 1000-Prompt Dataset..."
echo "================================================================================"
python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/merge_and_verify_oracle_dataset.py" \
  --input_dirs \
    "${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_2k" \
    "${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_1000" \
  --out_root "${DATASET_DIR}" \
  --total_prompts 1000 \
  --seeds 42 100 2024 \
  --primary_lambda "${PRIMARY_LAMBDA}"

echo ""
echo "================================================================================"
echo " [Step 2/3] Training & Benchmarking Router Models (Prompt-Disjoint Split)..."
echo "================================================================================"
python "${SCRIPT_DIR}/train_router.py" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}" \
  --model_type all \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --primary_lambda "${PRIMARY_LAMBDA}" \
  --seed "${SEED}"

echo ""
echo "================================================================================"
echo " [Step 3/3] Running Token Attribution & Semantic Keyword Analysis..."
echo "================================================================================"
python "${SCRIPT_DIR}/analyze_token_attribution.py" \
  --checkpoint "${OUT_DIR}/linear_ordinal_router.pt" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}/token_attribution" \
  --top_k 30

echo ""
echo "================================================================================"
echo " [Step 4/4] Printing Publication-Ready Master Benchmark Report..."
echo "================================================================================"
python "${SCRIPT_DIR}/print_results_summary.py" \
  --out_dir "${OUT_DIR}" \
  --dataset_dir "${DATASET_DIR}"

echo ""
echo "================================================================================"
echo " All Router Benchmarks and Interpretability Analyses Completed Successfully!"
echo " Check outputs in: ${OUT_DIR}"
echo "================================================================================"
