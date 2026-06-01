#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

MODE="${1:-all}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
STEPS="${STEPS:-1,2,3}"
OVERWRITE_LMDB="${OVERWRITE_LMDB:-0}"
OVERWRITE_X0PRED="${OVERWRITE_X0PRED:-0}"
CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG:-14b_cfgdistill_5k}"
CR_DISTILL_CLEAN_LMDB_DIR="${CR_DISTILL_CLEAN_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_5k}"

case "${MODE}" in
  all|generate|clean|x0pred)
    ;;
  *)
    echo "Usage: bash changing_resolution_distill/scripts/data/build_14b_cfgdistill_720p_clean_and_x0pred_lmdb_1k.sh [all|generate|clean|x0pred]" >&2
    exit 2
    ;;
esac

run_clean() {
  local clean_mode="$1"
  TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
  GPU_IDS="${GPU_IDS}" \
  OVERWRITE_LMDB="${OVERWRITE_LMDB}" \
  bash "${PROJECT_ROOT}/changing_resolution_distill/scripts/data/build_clean_480p720p_14b_cfgdistill_lmdb_1k_multigpu.sh" "${clean_mode}"
}

run_x0pred() {
  if [[ ! -d "${CR_DISTILL_CLEAN_LMDB_DIR}" ]] || [[ -z "$(find "${CR_DISTILL_CLEAN_LMDB_DIR}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
    echo "Clean 14B CfgDistill LMDB not found: ${CR_DISTILL_CLEAN_LMDB_DIR}" >&2
    echo "Run mode 'clean' or 'all' first." >&2
    exit 1
  fi

  IFS=',' read -r -a STEP_LIST <<< "${STEPS}"
  for step in "${STEP_LIST[@]}"; do
    step="$(echo "${step}" | xargs)"
    [[ -z "${step}" ]] && continue
    echo "===== build 14B CfgDistill x0_pred LMDB handoff step ${step} ====="
    HANDOFF_STEP="${step}" \
    TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
    GPU_IDS="${GPU_IDS}" \
    CR_STAGE2_LMDB_DIR="${CR_DISTILL_CLEAN_LMDB_DIR}" \
    CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG}" \
    OVERWRITE="${OVERWRITE_X0PRED}" \
    bash "${PROJECT_ROOT}/changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb_multigpu.sh"
  done
}

echo "14B CfgDistill full data rebuild"
echo "  project       : ${PROJECT_ROOT}"
echo "  mode          : ${MODE}"
echo "  total_samples : ${TOTAL_SAMPLES}"
echo "  gpu_ids       : ${GPU_IDS}"
echo "  steps         : ${STEPS}"
echo "  clean_lmdb    : ${CR_DISTILL_CLEAN_LMDB_DIR}"
echo "  stage3_tag    : ${CR_DISTILL_STAGE3_TAG}"

case "${MODE}" in
  generate)
    run_clean generate
    ;;
  clean)
    run_clean all
    ;;
  x0pred)
    run_x0pred
    ;;
  all)
    run_clean all
    run_x0pred
    ;;
esac
