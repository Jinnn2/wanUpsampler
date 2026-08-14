#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_uni_clean_train}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/changing_resolution_uni_clean_train.log}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/changing_resolution_uni/configs/train_universal_clean.yaml}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_uni_clean}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
RESUME="${RESUME:-}"

command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"; exit 0
fi

printf -v launch_command \
  'cd %q && CONFIG=%q DATA_DIR=%q OUT_DIR=%q CUDA_VISIBLE_DEVICES=%q NUM_GPUS=%q MAX_STEPS=%q RESUME=%q bash changing_resolution_uni/scripts/run_train.sh 2>&1 | tee %q' \
  "${PROJECT_ROOT}" "${CONFIG}" "${DATA_DIR}" "${OUT_DIR}" "${CUDA_VISIBLE_DEVICES}" \
  "${NUM_GPUS}" "${MAX_STEPS}" "${RESUME}" "${RUN_LOG}"
tmux new-session -d -s "${SESSION_NAME}" "${launch_command}"
echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${RUN_LOG}"
