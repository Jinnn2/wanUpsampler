#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

SESSION_NAME="${SESSION_NAME:-oracle_1500_resume_node${NODE_RANK:-0}}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-2}"
RUN_ID="${RUN_ID:-}"
NODE_NAME="${NODE_NAME:-node${NODE_RANK}}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PART_INDICES="${PART_INDICES:-}"
BASE_PROMPT_OFFSET="${BASE_PROMPT_OFFSET:-0}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-1000}"
VAL_PROMPTS="${VAL_PROMPTS:-200}"
TEST_PROMPTS="${TEST_PROMPTS:-300}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-42 100 2024}"
TRAIN_INCLUDE_NATIVE_HR="${TRAIN_INCLUDE_NATIVE_HR:-1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
PROMPTS_FILE="${PROMPTS_FILE:-${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_VIDEOS="${CLEAN_VIDEOS:-0}"
SAVE_LATENTS="${SAVE_LATENTS:-1}"
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE:-fp16}"
DRY_RUN="${DRY_RUN:-0}"
EXTRACT_T5="${EXTRACT_T5:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-604800}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"

[[ -n "${RUN_ID}" ]] || { echo "RUN_ID is required and must be identical on both nodes." >&2; exit 2; }
[[ "${NODE_RANK}" == "0" || "${NODE_RANK}" == "1" ]] || { echo "NODE_RANK must be 0 or 1." >&2; exit 2; }
if [[ -z "${PART_INDICES}" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then PART_INDICES="0,1,2,3"; else PART_INDICES="4,5,6,7"; fi
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed or not in PATH." >&2; exit 1; }
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists. Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/oracle_dataset_1500_16gpu/${RUN_ID}/${SESSION_NAME}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "Creating detached two-node resume session: ${SESSION_NAME}"
echo "  Run/Node   : ${RUN_ID} / rank ${NODE_RANK} (${NODE_NAME})"
echo "  GPUs/Parts : ${GPU_IDS} / ${PART_INDICES}"
echo "  Output     : ${OUT_ROOT}"
echo "  Log        : ${LOG_FILE}"

printf -v PIPELINE \
  'env PROJECT_ROOT=%q NODE_RANK=%q NUM_NODES=%q RUN_ID=%q NODE_NAME=%q GPU_IDS=%q PART_INDICES=%q BASE_PROMPT_OFFSET=%q TRAIN_PROMPTS=%q VAL_PROMPTS=%q TEST_PROMPTS=%q TRAIN_SEEDS=%q EVAL_SEEDS=%q TRAIN_INCLUDE_NATIVE_HR=%q OUT_ROOT=%q PROMPTS_FILE=%q PRIMARY_LAMBDA=%q SKIP_EXISTING=%q CLEAN_VIDEOS=%q SAVE_LATENTS=%q LATENT_SAVE_DTYPE=%q DRY_RUN=%q EXTRACT_T5=%q MIN_FREE_GIB=%q WAIT_INTERVAL=%q WAIT_TIMEOUT=%q MONITOR_INTERVAL=%q bash %q 2>&1 | tee -a %q' \
  "${PROJECT_ROOT}" "${NODE_RANK}" "${NUM_NODES}" "${RUN_ID}" "${NODE_NAME}" \
  "${GPU_IDS}" "${PART_INDICES}" "${BASE_PROMPT_OFFSET}" "${TRAIN_PROMPTS}" \
  "${VAL_PROMPTS}" "${TEST_PROMPTS}" "${TRAIN_SEEDS}" "${EVAL_SEEDS}" \
  "${TRAIN_INCLUDE_NATIVE_HR}" "${OUT_ROOT}" "${PROMPTS_FILE}" "${PRIMARY_LAMBDA}" \
  "${SKIP_EXISTING}" "${CLEAN_VIDEOS}" "${SAVE_LATENTS}" "${LATENT_SAVE_DTYPE}" \
  "${DRY_RUN}" "${EXTRACT_T5}" "${MIN_FREE_GIB}" "${WAIT_INTERVAL}" "${WAIT_TIMEOUT}" \
  "${MONITOR_INTERVAL}" "${SCRIPT_DIR}/build_oracle_dataset_1500_2gpu.sh" "${LOG_FILE}"
printf -v RUN_COMMAND 'bash -o pipefail -c %q' "${PIPELINE}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_COMMAND}"
echo "Started. Attach: tmux attach -t ${SESSION_NAME}"
echo "Logs: tail -f ${LOG_FILE}"
