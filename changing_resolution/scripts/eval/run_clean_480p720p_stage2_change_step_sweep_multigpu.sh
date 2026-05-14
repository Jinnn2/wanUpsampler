#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TOTAL_PROMPTS="${TOTAL_PROMPTS:-4}"
STEP_START="${STEP_START:-10}"
STEP_END="${STEP_END:-50}"
STEP_STRIDE="${STEP_STRIDE:-1}"
CHANGE_STEPS="${CHANGE_STEPS:-}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_stage2_change_step_sweep}"
OUT_DIR="${CR_STAGE2_CHANGE_STEP_SWEEP_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_stage2_change_step_sweep}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
if (( NUM_GPUS == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

if [[ -n "${CHANGE_STEPS}" ]]; then
  read -r -a steps <<< "${CHANGE_STEPS}"
else
  steps=()
  step="${STEP_START}"
  while (( step <= STEP_END )); do
    steps+=("${step}")
    step=$((step + STEP_STRIDE))
  done
fi
if [[ "${#steps[@]}" -eq 0 ]]; then
  echo "No change steps selected." >&2
  exit 1
fi

planned_videos=$((TOTAL_PROMPTS * ${#steps[@]}))
echo "Stage 2 change-step sweep:"
echo "  gpu_ids       : ${GPU_IDS}"
echo "  total_prompts : ${TOTAL_PROMPTS}"
echo "  change_steps  : ${steps[*]}"
echo "  planned panels: ${planned_videos}"
echo "  out_dir       : ${OUT_DIR}"
if [[ "${STEP_START}" == "10" && "${STEP_END}" == "50" && "${STEP_STRIDE}" == "1" && -z "${CHANGE_STEPS}" && "${TOTAL_PROMPTS}" == "4" ]]; then
  echo "Note: steps 10..50 inclusive gives 41 steps, so 4 prompts produce 164 panels, not 200." >&2
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

base_count=$((TOTAL_PROMPTS / NUM_GPUS))
remainder=$((TOTAL_PROMPTS % NUM_GPUS))
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
    STEP_START="${STEP_START}" \
    STEP_END="${STEP_END}" \
    STEP_STRIDE="${STEP_STRIDE}" \
    CHANGE_STEPS="${CHANGE_STEPS}" \
    bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh
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
  echo "Stage 2 change-step sweep failed. Check logs under: ${LOG_DIR}" >&2
  exit 1
fi

echo "Stage 2 change-step sweep ready: ${OUT_DIR}/compare"
