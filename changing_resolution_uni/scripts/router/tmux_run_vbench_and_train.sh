#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

SESSION_NAME="${SESSION_NAME:-vbench_score_and_train}"
LOG_FILE="${PROJECT_ROOT}/logs/vbench_and_train_${SESSION_NAME}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists. Kill with: tmux kill-session -t ${SESSION_NAME}" >&2
  echo "Or attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

echo "Creating detached tmux session: ${SESSION_NAME}"
echo "  Log File: ${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash -c 'bash \"${SCRIPT_DIR}/run_vbench_and_train.sh\" 2>&1 | tee -a \"${LOG_FILE}\"'"

echo "Started successfully in background!"
echo "To monitor live progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "To follow logs directly:"
echo "  tail -f ${LOG_FILE}"
