#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_uni_clean_train}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/changing_resolution_uni_clean_train.log}"

command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"; exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_ROOT}' && bash changing_resolution_uni/scripts/run_train.sh 2>&1 | tee '${RUN_LOG}'"
echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${RUN_LOG}"
