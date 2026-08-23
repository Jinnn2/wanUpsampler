#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

SESSION_NAME="${SESSION_NAME:-oracle_dataset_1500_8gpu}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PROMPT_OFFSET="${BASE_PROMPT_OFFSET:-0}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-1000}"
VAL_PROMPTS="${VAL_PROMPTS:-200}"
TEST_PROMPTS="${TEST_PROMPTS:-300}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-42 100 2024}"
TRAIN_INCLUDE_NATIVE_HR="${TRAIN_INCLUDE_NATIVE_HR:-1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_VIDEOS="${CLEAN_VIDEOS:-0}"
SAVE_LATENTS="${SAVE_LATENTS:-1}"
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE:-fp16}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRACT_T5="${EXTRACT_T5:-0}"
else
  EXTRACT_T5="${EXTRACT_T5:-1}"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH." >&2
  exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/oracle_dataset_1500_8gpu/${SESSION_NAME}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "Creating detached tmux session: ${SESSION_NAME}"
echo "  GPUs       : ${GPU_IDS}"
echo "  Split      : ${TRAIN_PROMPTS}/${VAL_PROMPTS}/${TEST_PROMPTS}"
echo "  Train seeds: ${TRAIN_SEEDS}"
echo "  Eval seeds : ${EVAL_SEEDS}"
echo "  Train native: ${TRAIN_INCLUDE_NATIVE_HR}"
echo "  Save latents: ${SAVE_LATENTS} (${LATENT_SAVE_DTYPE})"
echo "  Output     : ${OUT_ROOT}"
echo "  Log        : ${LOG_FILE}"

printf -v RUN_COMMAND \
  'env PROJECT_ROOT=%q GPU_IDS=%q BASE_PROMPT_OFFSET=%q TRAIN_PROMPTS=%q VAL_PROMPTS=%q TEST_PROMPTS=%q TRAIN_SEEDS=%q EVAL_SEEDS=%q TRAIN_INCLUDE_NATIVE_HR=%q OUT_ROOT=%q PROMPTS_FILE=%q PRIMARY_LAMBDA=%q SKIP_EXISTING=%q CLEAN_VIDEOS=%q SAVE_LATENTS=%q LATENT_SAVE_DTYPE=%q DRY_RUN=%q EXTRACT_T5=%q bash %q 2>&1 | tee -a %q' \
  "${PROJECT_ROOT}" "${GPU_IDS}" "${BASE_PROMPT_OFFSET}" "${TRAIN_PROMPTS}" \
  "${VAL_PROMPTS}" "${TEST_PROMPTS}" "${TRAIN_SEEDS}" "${EVAL_SEEDS}" \
  "${TRAIN_INCLUDE_NATIVE_HR}" "${OUT_ROOT}" "${PROMPTS_FILE}" "${PRIMARY_LAMBDA}" \
  "${SKIP_EXISTING}" "${CLEAN_VIDEOS}" "${SAVE_LATENTS}" "${LATENT_SAVE_DTYPE}" "${DRY_RUN}" "${EXTRACT_T5}" \
  "${SCRIPT_DIR}/build_oracle_dataset_1500_8gpu.sh" "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_COMMAND}"

echo "Started successfully."
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Logs  : tail -f ${LOG_FILE}"
