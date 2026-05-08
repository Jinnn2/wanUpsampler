#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SESSION="${TMUX_SESSION:-wan_cr_clean_480p720p}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SESSION}.log}"

mkdir -p "${LOG_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -c "${PROJECT_ROOT}" \
  "bash -lc 'set -euo pipefail; \
    echo \"[start] \$(date)\"; \
    echo \"[cwd] \$(pwd)\"; \
    bash changing_resolution/scripts/legacy/run_clean_480p720p_training.sh all 2>&1 | tee -a \"${LOG_FILE}\"; \
    status=\${PIPESTATUS[0]}; \
    echo \"[end] \$(date) status=\${status}\" | tee -a \"${LOG_FILE}\"; \
    exit \${status}'"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG_FILE}"
