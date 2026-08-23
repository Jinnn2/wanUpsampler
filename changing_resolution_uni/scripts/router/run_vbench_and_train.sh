#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
SOURCE_DATASET_DIRS="${SOURCE_DATASET_DIRS:-${DATASET_DIR}}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
NGPUS="${NGPUS:-4}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-1000}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-42 100 2024}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
DIAGNOSTIC_DIMENSIONS="${DIAGNOSTIC_DIMENSIONS:-}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_benchmarks_1k_lambda$(echo "${PRIMARY_LAMBDA}" | tr -d '.')}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
read -r -a SOURCE_DATASET_ARRAY <<< "${SOURCE_DATASET_DIRS}"

echo "Dataset target : ${DATASET_DIR}"
echo "Dataset sources: ${SOURCE_DATASET_DIRS}"
echo "Expected data  : ${EXPECTED_PROMPTS} prompts x seeds [${EXPECTED_SEEDS}]"
echo "Primary lambda : ${PRIMARY_LAMBDA}"
echo "Diagnostics    : ${DIAGNOSTIC_DIMENSIONS:-none}"
echo "Router outputs : ${OUT_DIR}"

# Check and auto-install all dependencies and pre-warm models if needed
if ! python -c "import clip, pyiqa; assert hasattr(clip, 'load')" 2>/dev/null; then
  echo "Setting up missing environment dependencies & pre-warming models..."
  bash "${SCRIPT_DIR}/setup_environment.sh"
fi

echo "================================================================================"
echo " [Step 1/5] Running Distributed VBench-5 Quality Scoring on Generated Videos..."
echo "================================================================================"
score_args=(
  --input_dirs "${SOURCE_DATASET_ARRAY[@]}"
  --dataset_dir "${DATASET_DIR}"
  --vbench_root "${VBENCH_ROOT}"
  --ngpus "${NGPUS}"
  --expected_prompts "${EXPECTED_PROMPTS}"
  --expected_seeds ${EXPECTED_SEEDS}
  --primary_lambda "${PRIMARY_LAMBDA}"
)
if [[ -n "${DIAGNOSTIC_DIMENSIONS}" ]]; then
  read -r -a diagnostic_dimension_array <<< "${DIAGNOSTIC_DIMENSIONS}"
  score_args+=(--diagnostic_dimensions "${diagnostic_dimension_array[@]}")
fi
if [[ "${FORCE_RESCORE}" == "1" ]]; then
  score_args+=(--force_rescore)
fi
if [[ -n "${EXPECTED_VBENCH_COMMIT}" ]]; then
  score_args+=(--expected_vbench_commit "${EXPECTED_VBENCH_COMMIT}")
fi
python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/batch_vbench_score_dataset.py" "${score_args[@]}"

echo ""
echo "================================================================================"
echo " [Step 2/5] Strictly auditing scored oracle records..."
echo "================================================================================"
python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/cleanup_legacy_records.py" \
  --dataset_dir "${DATASET_DIR}" \
  --profile formal \
  --strict

echo ""
echo "================================================================================"
echo " [Step 3/5] Training & Benchmarking Router Models on prompt-level labels..."
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
echo " [Step 4/5] Running Token Attribution & Semantic Keyword Analysis..."
echo "================================================================================"
python "${SCRIPT_DIR}/analyze_token_attribution.py" \
  --checkpoint "${OUT_DIR}/linear_ordinal_router.pt" \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}/token_attribution" \
  --top_k 30

echo ""
echo "================================================================================"
echo " [Step 5/5] Printing Comprehensive Publication-Ready Report..."
echo "================================================================================"
python "${SCRIPT_DIR}/print_results_summary.py" \
  --out_dir "${OUT_DIR}" \
  --dataset_dir "${DATASET_DIR}"

echo ""
echo "================================================================================"
echo " Pipeline Complete! Results saved to: ${OUT_DIR}"
echo "================================================================================"
