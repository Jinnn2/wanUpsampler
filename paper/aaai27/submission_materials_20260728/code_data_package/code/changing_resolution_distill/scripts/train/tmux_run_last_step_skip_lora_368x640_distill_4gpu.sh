#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_distill_lora_368x640_4gpu}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/train_distill_lora_368x640_4gpu.log}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

command -v tmux >/dev/null 2>&1 || { echo "tmux not found." >&2; exit 1; }
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_ROOT}' && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' bash changing_resolution_distill/scripts/train/run_last_step_skip_lora_368x640_distill_4gpu.sh train 2>&1 | tee '${RUN_LOG}'"

echo "Started distill 368x640 LoRA four-GPU training: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${RUN_LOG}"
