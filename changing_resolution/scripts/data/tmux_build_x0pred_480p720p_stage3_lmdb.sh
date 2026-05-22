#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DENOISE_STEP="${DENOISE_STEP:-45}"
SESSION_NAME="${SESSION_NAME:-wan_cr_stage3_x0pred_lmdb_step${DENOISE_STEP}_build}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/build_x0pred_480p720p_stage3_lmdb_step${DENOISE_STEP}.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run build_x0pred_480p720p_stage3_lmdb.sh directly." >&2
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
  "cd '${PROJECT_ROOT}' && DENOISE_STEP='${DENOISE_STEP}' bash changing_resolution/scripts/data/build_x0pred_480p720p_stage3_lmdb.sh 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${RUN_LOG}"
