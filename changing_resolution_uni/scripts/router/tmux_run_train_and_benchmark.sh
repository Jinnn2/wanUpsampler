#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_benchmarks_1k_lambda$(echo "${PRIMARY_LAMBDA}" | tr -d '.')}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"

SESSION_NAME="${SESSION_NAME:-router_train}"
LOG_FILE="${PROJECT_ROOT}/logs/train_${SESSION_NAME}.log"
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

printf -v RUN_COMMAND \
  'env PROJECT_ROOT=%q DATASET_DIR=%q PRIMARY_LAMBDA=%q OUT_DIR=%q EPOCHS=%q BATCH_SIZE=%q LR=%q SEED=%q bash %q 2>&1 | tee -a %q' \
  "${PROJECT_ROOT}" "${DATASET_DIR}" "${PRIMARY_LAMBDA}" "${OUT_DIR}" \
  "${EPOCHS}" "${BATCH_SIZE}" "${LR}" "${SEED}" \
  "${SCRIPT_DIR}/run_train_and_benchmark.sh" "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_COMMAND}"

echo "Started successfully in background!"
echo "To monitor live progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "To follow logs directly:"
echo "  tail -f ${LOG_FILE}"
