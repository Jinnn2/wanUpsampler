#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_benchmarks_1k_lambda$(echo "${PRIMARY_LAMBDA}" | tr -d '.')}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "================================================================================"
echo "          PROMPT-CONDITIONED TIMESTEP ROUTER RETRAINING PIPELINE"
echo "================================================================================"
echo "  Dataset Directory : ${DATASET_DIR}"
echo "  Output Directory  : ${OUT_DIR}"
echo "  Primary Lambda    : ${PRIMARY_LAMBDA}"
echo "  Epochs / BatchSize: ${EPOCHS} / ${BATCH_SIZE}"
echo "================================================================================"

# ── Step 1: Clean & Prepare Dataset Records ────────────────────────────────────
echo ""
echo "[Step 1/4] Verifying and cleaning trajectory records in ${DATASET_DIR}..."
if [[ -f "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/cleanup_legacy_records.py" ]]; then
  python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/cleanup_legacy_records.py"
fi

# ── Step 2: Train & Benchmark All Router Models ────────────────────────────────
echo ""
echo "[Step 2/4] Training & Benchmarking Router Models (Prompt-Disjoint Split)..."
python "${SCRIPT_DIR}/train_router.py" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}" \
  --model_type all \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --primary_lambda "${PRIMARY_LAMBDA}" \
  --seed "${SEED}"

# ── Step 3: Reverse Token Attribution & Semantic Keyword Extraction ───────────
echo ""
echo "[Step 3/4] Running Reverse Token Attribution & Semantic Discovery..."
python "${SCRIPT_DIR}/analyze_token_attribution.py" \
  --checkpoint "${OUT_DIR}/linear_ordinal_router.pt" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}/token_attribution" \
  --top_k 30

# ── Step 4: Print Master Benchmark Report ─────────────────────────────────────
echo ""
echo "[Step 4/4] Generating Publication-Ready Master Benchmark Report..."
python "${SCRIPT_DIR}/print_results_summary.py" \
  --out_dir "${OUT_DIR}" \
  --dataset_dir "${DATASET_DIR}"

echo ""
echo "================================================================================"
echo " Retraining Pipeline Complete! All artifacts saved to: ${OUT_DIR}"
echo "================================================================================"
