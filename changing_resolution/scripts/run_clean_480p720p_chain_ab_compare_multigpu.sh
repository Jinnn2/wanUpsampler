#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-16}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_chain_ab_compare}"
OUT_DIR="${CR_CHAIN_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_chain_ab_stage1}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

base_count=$((TOTAL_SAMPLES / NUM_GPUS))
remainder=$((TOTAL_SAMPLES % NUM_GPUS))
offset=0
pids=()

for rank in "${!GPUS[@]}"; do
  count="${base_count}"
  if (( rank < remainder )); then
    count=$((count + 1))
  fi
  if (( count == 0 )); then
    continue
  fi

  gpu="${GPUS[$rank]}"
  part_name="$(printf "part_%02d" "${rank}")"
  log_path="${LOG_DIR}/${part_name}.log"
  echo "Launch ${part_name}: gpu=${gpu}, prompt_offset=${offset}, count=${count}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PROMPT_OFFSET="${offset}" \
    LIMIT="${count}" \
    bash changing_resolution/scripts/run_clean_480p720p_chain_ab_compare.sh
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  offset=$((offset + count))
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "Chain A/B compare failed. Check logs under: ${LOG_DIR}" >&2
  exit 1
fi

echo "Chain A/B compare ready: ${OUT_DIR}/compare"
