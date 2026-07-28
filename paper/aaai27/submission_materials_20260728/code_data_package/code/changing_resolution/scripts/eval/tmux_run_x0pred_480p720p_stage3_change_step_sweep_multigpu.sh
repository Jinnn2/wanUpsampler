#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_stage3_step_sweep_4gpu}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/stage3_change_step_sweep_4gpu.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run run_x0pred_480p720p_stage3_change_step_sweep_multigpu.sh directly." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Log: ${RUN_LOG}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_ROOT}' && GPU_IDS='${GPU_IDS:-0,1,2,3}' TOTAL_PROMPTS='${TOTAL_PROMPTS:-4}' STEP_START='${STEP_START:-10}' STEP_END='${STEP_END:-50}' STEP_STRIDE='${STEP_STRIDE:-1}' CHANGE_STEPS='${CHANGE_STEPS:-}' bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep_multigpu.sh 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${RUN_LOG}"
