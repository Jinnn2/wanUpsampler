#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

SESSION_NAME="${SESSION_NAME:-oracle_1500_h100_tail8}"
RUN_ID="${RUN_ID:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
PROMPTS_FILE="${PROMPTS_FILE:-${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}}"
MICRO_BATCH_PROMPTS="${MICRO_BATCH_PROMPTS:-2}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE:-fp16}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
CONFIRM_EXCLUSIVE="${CONFIRM_EXCLUSIVE:-0}"

[[ -n "${RUN_ID}" ]] || { echo "RUN_ID is required." >&2; exit 2; }
[[ "${CONFIRM_EXCLUSIVE}" == "1" ]] || {
  echo "Set CONFIRM_EXCLUSIVE=1 only after every previous writer has stopped." >&2
  exit 2
}
command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed." >&2; exit 1; }
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/oracle_dataset_1500_h100_tail/${RUN_ID}/${SESSION_NAME}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

printf -v PIPELINE \
  'env PROJECT_ROOT=%q RUN_ID=%q GPU_IDS=%q OUT_ROOT=%q PROMPTS_FILE=%q MICRO_BATCH_PROMPTS=%q PRIMARY_LAMBDA=%q LATENT_SAVE_DTYPE=%q MONITOR_INTERVAL=%q MIN_FREE_GIB=%q CONFIRM_EXCLUSIVE=%q bash %q 2>&1 | tee -a %q' \
  "${PROJECT_ROOT}" "${RUN_ID}" "${GPU_IDS}" "${OUT_ROOT}" "${PROMPTS_FILE}" \
  "${MICRO_BATCH_PROMPTS}" "${PRIMARY_LAMBDA}" "${LATENT_SAVE_DTYPE}" \
  "${MONITOR_INTERVAL}" "${MIN_FREE_GIB}" "${CONFIRM_EXCLUSIVE}" \
  "${SCRIPT_DIR}/build_oracle_dataset_1500_h100_tail8.sh" "${LOG_FILE}"
printf -v RUN_COMMAND 'bash -o pipefail -c %q' "${PIPELINE}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_COMMAND}"
echo "Started H100 tail session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
