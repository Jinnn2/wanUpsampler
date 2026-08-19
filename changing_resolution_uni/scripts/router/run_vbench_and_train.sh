#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
NGPUS="${NGPUS:-4}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.05}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_benchmarks_1k}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Clean potential corrupted torch hub cache from previous multi-process race
rm -rf "${HOME}/.cache/torch/hub/main.zip"* "${HOME}/.cache/torch/hub/facebookresearch-dino"* "${HOME}/.cache/torch/hub/facebookresearch_dino"* 2>/dev/null || true

echo "================================================================================"
echo " [Step 1/3] Running Distributed VBench-5 Quality Scoring on Generated Videos..."
echo "================================================================================"
python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/batch_vbench_score_dataset.py" \
  --input_dirs \
    "${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_2k" \
    "${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_500_1000" \
    "${DATASET_DIR}" \
  --dataset_dir "${DATASET_DIR}" \
  --vbench_root "${VBENCH_ROOT}" \
  --ngpus "${NGPUS}" \
  --primary_lambda "${PRIMARY_LAMBDA}"

echo ""
echo "================================================================================"
echo " [Step 2/3] Training & Benchmarking Router Models on Genuine Oracle Labels..."
echo "================================================================================"
python "${SCRIPT_DIR}/train_router.py" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}" \
  --model_type all \
  --epochs 40 \
  --batch_size 32 \
  --lr 0.001 \
  --primary_lambda "${PRIMARY_LAMBDA}"

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
echo " Pipeline Complete! Results saved to: ${OUT_DIR}"
echo "================================================================================"
