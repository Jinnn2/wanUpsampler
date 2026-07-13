#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_stage2_368x640_4gpu}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/train_clean_368x640_720x1248_stage2_4gpu.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Run run_clean_368x640_720x1248_stage2_lmdb_training.sh directly." >&2
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
  "cd '${PROJECT_ROOT}' && NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash changing_resolution/scripts/train/run_clean_368x640_720x1248_stage2_lmdb_training.sh train 2>&1 | tee '${RUN_LOG}'"

echo "Started 4-GPU Stage2 training: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${RUN_LOG}"
