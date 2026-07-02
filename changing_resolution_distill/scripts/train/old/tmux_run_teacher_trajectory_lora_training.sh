#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SESSION="${SESSION:-cr_teacher_traj_lora}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SESSION}.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found; run directly:" >&2
  echo "bash ${SCRIPT_DIR}/run_teacher_trajectory_lora_training.sh train" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && bash '${SCRIPT_DIR}/run_teacher_trajectory_lora_training.sh' train 2>&1 | tee '${LOG_FILE}'"

echo "Started tmux session: ${SESSION}"
echo "Log: ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION}"
