#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
SOURCE_DATASET_DIRS="${SOURCE_DATASET_DIRS:-${DATASET_DIR}}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
NGPUS="${NGPUS:-4}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-1000}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-42 100 2024}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_benchmarks_1k_lambda$(echo "${PRIMARY_LAMBDA}" | tr -d '.')}"

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

printf -v RUN_COMMAND \
  'env PROJECT_ROOT=%q DATASET_DIR=%q SOURCE_DATASET_DIRS=%q VBENCH_ROOT=%q NGPUS=%q EXPECTED_PROMPTS=%q EXPECTED_SEEDS=%q PRIMARY_LAMBDA=%q OUT_DIR=%q bash %q 2>&1 | tee -a %q' \
  "${PROJECT_ROOT}" "${DATASET_DIR}" "${SOURCE_DATASET_DIRS}" "${VBENCH_ROOT}" "${NGPUS}" \
  "${EXPECTED_PROMPTS}" "${EXPECTED_SEEDS}" "${PRIMARY_LAMBDA}" "${OUT_DIR}" \
  "${SCRIPT_DIR}/run_vbench_and_train.sh" "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_COMMAND}"

echo "Started successfully in background!"
echo "To monitor live progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "To follow logs directly:"
echo "  tail -f ${LOG_FILE}"
