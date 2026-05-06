#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-wan_cr_lmdb_480p720p_1k}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs_2/houze/wanUpsampler}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_ROOT}' && bash changing_resolution/scripts/build_clean_480p720p_lmdb_1k.sh all 2>&1 | tee '${LOG_DIR}/build_clean_lmdb_480p720p_1k.log'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${LOG_DIR}/build_clean_lmdb_480p720p_1k.log"
