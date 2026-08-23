#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

SESSION_NAME="${SESSION_NAME:-oracle_dataset_2k_4gpu}"
TOTAL_PROMPTS="${TOTAL_PROMPTS:-2000}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEEDS="${SEEDS:-42 100 2024}"
OUT_ROOT="${OUT_ROOT:-}"
CLEAN_VIDEOS="${CLEAN_VIDEOS:-0}"
DRY_RUN="${DRY_RUN:-0}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/oracle_dataset_4gpu/${SESSION_NAME}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "Creating detached tmux session: ${SESSION_NAME}"
echo "  Prompt Offset: ${PROMPT_OFFSET}"
echo "  Total Prompts: ${TOTAL_PROMPTS}"
echo "  GPU Devices  : ${GPU_IDS}"
echo "  Seeds        : ${SEEDS}"
echo "  Primary Lambda: ${PRIMARY_LAMBDA}"
echo "  Log File     : ${LOG_FILE}"

printf -v RUN_COMMAND \
  'env PROJECT_ROOT=%q TOTAL_PROMPTS=%q PROMPT_OFFSET=%q GPU_IDS=%q SEEDS=%q OUT_ROOT=%q CLEAN_VIDEOS=%q DRY_RUN=%q PRIMARY_LAMBDA=%q bash %q 2>&1 | tee -a %q' \
  "${PROJECT_ROOT}" "${TOTAL_PROMPTS}" "${PROMPT_OFFSET}" "${GPU_IDS}" \
  "${SEEDS}" "${OUT_ROOT}" "${CLEAN_VIDEOS}" "${DRY_RUN}" \
  "${PRIMARY_LAMBDA}" "${SCRIPT_DIR}/build_oracle_dataset_4gpu.sh" "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_COMMAND}"

echo "Started successfully!"
echo "To attach to monitor progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "To follow logs directly:"
echo "  tail -f ${LOG_FILE}"
