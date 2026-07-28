#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GROUP="${1:-factorial}"
SESSION_NAME="${SESSION_NAME:-aaai27_${GROUP}}"
LOG_DIR="${AAAI_LOG_DIR:-${PROJECT_ROOT}/outputs/aaai27_experiments/_state}"

command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
log_path="${LOG_DIR}/${SESSION_NAME}.console.log"
command="cd '${PROJECT_ROOT}' && python paper/aaai27/experiments/run_experiments.py run --group '${GROUP}' --keep-going 2>&1 | tee -a '${log_path}'"
tmux new-session -d -s "${SESSION_NAME}" "bash -lc \"${command}\""

echo "Started: ${SESSION_NAME}"
echo "Log    : ${log_path}"
echo "Attach : tmux attach -t ${SESSION_NAME}"
