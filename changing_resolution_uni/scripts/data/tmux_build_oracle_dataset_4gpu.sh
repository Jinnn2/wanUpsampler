#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

SESSION_NAME="${SESSION_NAME:-oracle_dataset_2k_4gpu}"
TOTAL_PROMPTS="${TOTAL_PROMPTS:-2000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEEDS="${SEEDS:-42 100 2024}"
DRY_RUN="${DRY_RUN:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/oracle_dataset_4gpu/tmux_master.log"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "Creating detached tmux session: ${SESSION_NAME}"
echo "  Total Prompts: ${TOTAL_PROMPTS}"
echo "  GPU Devices  : ${GPU_IDS}"
echo "  Seeds        : ${SEEDS}"
echo "  Log File     : ${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash -c 'TOTAL_PROMPTS=\"${TOTAL_PROMPTS}\" GPU_IDS=\"${GPU_IDS}\" SEEDS=\"${SEEDS}\" DRY_RUN=\"${DRY_RUN}\" bash \"${SCRIPT_DIR}/build_oracle_dataset_4gpu.sh\" 2>&1 | tee -a \"${LOG_FILE}\"'"

echo "Started successfully!"
echo "To attach to monitor progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "To follow logs directly:"
echo "  tail -f ${LOG_FILE}"
