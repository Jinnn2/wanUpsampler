#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  check|prepare|score|status|finalize|merge|embed|train|train-b4|train-prior-v2|all) ;;
  *)
    echo "Usage: $0 [check|prepare|score|status|finalize|merge|embed|train|train-b4|train-prior-v2|all]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/envs/vbench/bin/python}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
PRIMARY_ROOT="${PRIMARY_ROOT:-${PROJECT_ROOT}/outputs/univ_low_budget_extension_primary_v1}"
RESERVE_ROOT="${RESERVE_ROOT:-${PROJECT_ROOT}/outputs/univ_low_budget_extension_reserve_v1}"
SCORE_ROOT="${SCORE_ROOT:-${PROJECT_ROOT}/outputs/univ_combined_v3_scoring_v1}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/outputs/univ_combined_v3_trainval_v1}"
TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_combined_v3_budget_prior_v1}"
B4_OUT_ROOT="${B4_OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_combined_v3_b4_control_v1}"
PRIOR_V2_OUT_ROOT="${PRIOR_V2_OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_combined_v3_prompt_prior_v2}"
QUALITY_CURVE_ROOT="${QUALITY_CURVE_ROOT:-${TRAIN_OUT_ROOT}}"
VBENCH_NGPUS="${VBENCH_NGPUS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-}"
DIAGNOSTIC_DIMENSIONS="${DIAGNOSTIC_DIMENSIONS:-dynamic_degree}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
SKIP_VBENCH_WARMUP="${SKIP_VBENCH_WARMUP:-0}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 100 2024}"
LAMBDAS="${LAMBDAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
B4_EPOCHS="${B4_EPOCHS:-40}"
B4_BATCH_SIZE="${B4_BATCH_SIZE:-32}"
B4_LR="${B4_LR:-0.001}"
B4_WEIGHT_DECAY="${B4_WEIGHT_DECAY:-0.0001}"
B4_SOFT_TARGET_TAU="${B4_SOFT_TARGET_TAU:-0.02}"
B4_EMD_WEIGHT="${B4_EMD_WEIGHT:-0.5}"
PRIOR_V2_TRAIN_SEEDS="${PRIOR_V2_TRAIN_SEEDS:-42 100 2024 31415 27182}"
PRIOR_V2_MAX_EPOCHS="${PRIOR_V2_MAX_EPOCHS:-30}"
PRIOR_V2_BATCH_SIZE="${PRIOR_V2_BATCH_SIZE:-32}"
PRIOR_V2_CV_FOLDS="${PRIOR_V2_CV_FOLDS:-5}"
HARDWARE_LABEL="${HARDWARE_LABEL:-unspecified_generation_device}"

SCORER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/score_combined_v3_dataset.py"
MERGER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/merge_scored_combined_v3.py"
EMBEDDER="${PROJECT_ROOT}/changing_resolution_uni/scripts/data/extract_prompt_t5_embeddings.py"
TRAINER="${PROJECT_ROOT}/UNIV_adaptor/scripts/router/train_combined_v3_budget_prior.py"
B4_TRAINER="${PROJECT_ROOT}/UNIV_adaptor/scripts/router/train_combined_v3_b4_control.py"
PRIOR_V2_TRAINER="${PROJECT_ROOT}/UNIV_adaptor/scripts/router/train_combined_v3_prompt_prior_v2.py"
SCORE_MANIFEST="${SCORE_ROOT}/score_manifest.json"
SCORED_MANIFEST="${SCORE_ROOT}/scored_dataset_manifest.json"

require_file() {
  [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }
}

resolve_vbench_python() {
  if [[ -x "${VBENCH_PYTHON}" ]] && (
    cd "${VBENCH_ROOT}" && "${VBENCH_PYTHON}" -c 'import torch, vbench' >/dev/null 2>&1
  ); then
    return
  fi
  local candidate
  for candidate in /opt/conda/envs/vbench/bin/python /opt/conda/bin/python; do
    if [[ -x "${candidate}" ]] && (
      cd "${VBENCH_ROOT}" && "${candidate}" -c 'import torch, vbench' >/dev/null 2>&1
    ); then
      VBENCH_PYTHON="${candidate}"
      return
    fi
  done
  echo "No Python environment can import torch and vbench from ${VBENCH_ROOT}." >&2
  exit 1
}

resolve_vbench_commit() {
  if [[ -z "${EXPECTED_VBENCH_COMMIT}" ]]; then
    EXPECTED_VBENCH_COMMIT="$(git -C "${VBENCH_ROOT}" rev-parse HEAD)"
  fi
}

check_inputs() {
  [[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
  for path in "${SCORER}" "${MERGER}" "${EMBEDDER}" "${TRAINER}" "${B4_TRAINER}" "${PRIOR_V2_TRAINER}"; do
    require_file "${path}"
  done
  for root in "${PRIMARY_ROOT}" "${RESERVE_ROOT}"; do
    require_file "${root}/extension_manifest.json"
    [[ "$(find "${root}/combined_records/train" -maxdepth 1 -name '*.json' | wc -l)" -eq 300 ]] || {
      echo "Expected 300 train records under ${root}" >&2
      exit 1
    }
    [[ "$(find "${root}/combined_records/validation" -maxdepth 1 -name '*.json' | wc -l)" -eq 300 ]] || {
      echo "Expected 300 validation records under ${root}" >&2
      exit 1
    }
  done
  require_dir "${VBENCH_ROOT}"
  require_file "${VBENCH_ROOT}/evaluate.py"
  require_dir "${MODEL_ROOT}"
  require_file "${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
  require_dir "${MODEL_ROOT}/google/umt5-xxl"
  require_dir "${LIGHTX2V_REPO}"
  resolve_vbench_python
  resolve_vbench_commit
  [[ "${VBENCH_NGPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "VBENCH_NGPUS must be positive." >&2; exit 2; }
  [[ "${FORCE_RESCORE}" == "0" || "${FORCE_RESCORE}" == "1" ]] || { echo "FORCE_RESCORE must be 0 or 1." >&2; exit 2; }
  (
    cd "${VBENCH_ROOT}"
    "${VBENCH_PYTHON}" -c 'import torch, vbench; assert torch.cuda.is_available()'
  )
  PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${WAN_PYTHON}" -c 'import numpy, torch; import UNIV_adaptor.combined_v3; from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel; print("LightX2V native T5 import passed:", T5EncoderModel.__module__)'
  echo "Combined-v3 inputs passed"
  echo "VBench commit: ${EXPECTED_VBENCH_COMMIT}"
  echo "Selection splits: train validation (test excluded)"
}

prepare_scoring() {
  "${WAN_PYTHON}" "${SCORER}" prepare \
    --shard "primary=${PRIMARY_ROOT}" \
    --shard "reserve=${RESERVE_ROOT}" \
    --out-root "${SCORE_ROOT}" \
    --splits train validation
}

score_all_cases() {
  require_file "${SCORE_MANIFEST}"
  resolve_vbench_python
  resolve_vbench_commit
  local lock="${SCORE_ROOT}/.vbench_scoring.lock"
  if ! mkdir "${lock}" 2>/dev/null; then
    echo "Scoring lock exists: ${lock}" >&2
    exit 1
  fi
  cleanup() { rmdir "${lock}" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
  mkdir -p "${SCORE_ROOT}/logs"
  if [[ "${SKIP_VBENCH_WARMUP}" != "1" ]]; then
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${VBENCH_PYTHON}" - "${VBENCH_PYTHON}" "${VBENCH_ROOT}" <<'PY'
import sys
from pathlib import Path

from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
    warmup_vbench_cache,
)

warmup_vbench_cache(sys.argv[1], Path(sys.argv[2]))
PY
  fi
  local -a diagnostic_args=()
  if [[ -n "${DIAGNOSTIC_DIMENSIONS}" ]]; then
    read -r -a diagnostic_args <<< "${DIAGNOSTIC_DIMENSIONS}"
  fi
  local case_id log
  while IFS= read -r case_id; do
    [[ -n "${case_id}" ]] || continue
    log="${SCORE_ROOT}/logs/${case_id}.log"
    echo "[score] ${case_id} -> ${log}"
    local -a args=(
      "${SCORER}" score-case
      --manifest "${SCORE_MANIFEST}"
      --case-id "${case_id}"
      --vbench-root "${VBENCH_ROOT}"
      --vbench-python "${VBENCH_PYTHON}"
      --ngpus "${VBENCH_NGPUS}"
      --expected-vbench-commit "${EXPECTED_VBENCH_COMMIT}"
      --diagnostic-dimensions "${diagnostic_args[@]}"
    )
    [[ "${FORCE_RESCORE}" == "1" ]] && args+=(--force-rescore)
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${VBENCH_PYTHON}" "${args[@]}" 2>&1 | tee -a "${log}"
  done < <("${WAN_PYTHON}" "${SCORER}" list-cases --manifest "${SCORE_MANIFEST}")
  cleanup
  trap - EXIT INT TERM
}

finalize_scores() {
  "${WAN_PYTHON}" "${SCORER}" finalize --manifest "${SCORE_MANIFEST}"
}

merge_dataset() {
  require_file "${SCORED_MANIFEST}"
  "${WAN_PYTHON}" "${MERGER}" \
    --scored-manifest "${SCORED_MANIFEST}" \
    --output-root "${DATASET_ROOT}" \
    --expected-train-prompts 600 \
    --expected-validation-prompts 200 \
    --validation-seeds 3
}

embed_prompts() {
  require_file "${DATASET_ROOT}/dataset_index.json"
  require_file "${DATASET_ROOT}/prompts.txt"
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
    LIGHTX2V_REPO="${LIGHTX2V_REPO}" \
    PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${WAN_PYTHON}" "${EMBEDDER}" \
      --prompts_file "${DATASET_ROOT}/prompts.txt" \
      --out_dir "${DATASET_ROOT}/t5_embeddings" \
      --model_path "${MODEL_ROOT}" \
      --device cuda \
      --precision bf16 \
      --required_backend wan_native \
      --skip_existing
}

train_prior() {
  require_file "${DATASET_ROOT}/t5_embeddings/t5_manifest.json"
  read -r -a train_seed_args <<< "${TRAIN_SEEDS}"
  read -r -a lambda_args <<< "${LAMBDAS}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${WAN_PYTHON}" "${TRAINER}" \
      --dataset-root "${DATASET_ROOT}" \
      --out-root "${TRAIN_OUT_ROOT}" \
      --train-seeds "${train_seed_args[@]}" \
      --lambdas "${lambda_args[@]}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --hardware-label "${HARDWARE_LABEL}" \
      --device cuda
}

train_b4_control() {
  require_file "${DATASET_ROOT}/t5_embeddings/t5_manifest.json"
  require_file "${QUALITY_CURVE_ROOT}/selection_summary.json"
  read -r -a train_seed_args <<< "${TRAIN_SEEDS}"
  read -r -a lambda_args <<< "${LAMBDAS}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${WAN_PYTHON}" "${B4_TRAINER}" \
      --dataset-root "${DATASET_ROOT}" \
      --quality-curve-root "${QUALITY_CURVE_ROOT}" \
      --out-root "${B4_OUT_ROOT}" \
      --train-seeds "${train_seed_args[@]}" \
      --lambdas "${lambda_args[@]}" \
      --epochs "${B4_EPOCHS}" \
      --batch-size "${B4_BATCH_SIZE}" \
      --lr "${B4_LR}" \
      --weight-decay "${B4_WEIGHT_DECAY}" \
      --soft-target-tau "${B4_SOFT_TARGET_TAU}" \
      --emd-weight "${B4_EMD_WEIGHT}" \
      --hardware-label "${HARDWARE_LABEL}" \
      --device cuda
}

train_prompt_prior_v2() {
  require_file "${DATASET_ROOT}/t5_embeddings/t5_manifest.json"
  require_file "${B4_OUT_ROOT}/selection_summary.json"
  read -r -a train_seed_args <<< "${PRIOR_V2_TRAIN_SEEDS}"
  read -r -a lambda_args <<< "${LAMBDAS}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${WAN_PYTHON}" "${PRIOR_V2_TRAINER}" \
      --dataset-root "${DATASET_ROOT}" \
      --b4-root "${B4_OUT_ROOT}" \
      --out-root "${PRIOR_V2_OUT_ROOT}" \
      --train-seeds "${train_seed_args[@]}" \
      --lambdas "${lambda_args[@]}" \
      --cv-folds "${PRIOR_V2_CV_FOLDS}" \
      --max-epochs "${PRIOR_V2_MAX_EPOCHS}" \
      --batch-size "${PRIOR_V2_BATCH_SIZE}" \
      --hardware-label "${HARDWARE_LABEL}" \
      --device cuda
}

case "${MODE}" in
  check) check_inputs ;;
  prepare) prepare_scoring ;;
  score) score_all_cases ;;
  status) "${WAN_PYTHON}" "${SCORER}" status --manifest "${SCORE_MANIFEST}" ;;
  finalize) finalize_scores ;;
  merge) merge_dataset ;;
  embed) embed_prompts ;;
  train) train_prior ;;
  train-b4) train_b4_control ;;
  train-prior-v2) train_prompt_prior_v2 ;;
  all)
    check_inputs
    prepare_scoring
    score_all_cases
    finalize_scores
    merge_dataset
    embed_prompts
    train_prior
    ;;
esac

echo "Score root   : ${SCORE_ROOT}"
echo "Dataset root : ${DATASET_ROOT}"
echo "Training root: ${TRAIN_OUT_ROOT}"
echo "B4 root      : ${B4_OUT_ROOT}"
echo "Prior V2 root: ${PRIOR_V2_OUT_ROOT}"
