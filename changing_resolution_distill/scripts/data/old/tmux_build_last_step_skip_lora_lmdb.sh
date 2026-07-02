#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_distill_last_step_skip_lora_build}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
START_OFFSET="${START_OFFSET:-0}"
OVERWRITE="${OVERWRITE:-0}"
TMUX_LOG_DIR="${TMUX_LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${TMUX_LOG_DIR}/build_last_step_skip_lora_lmdb.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${TMUX_LOG_DIR}/run_last_step_skip_lora_lmdb_build.tmux.sh}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run the multigpu build script directly." >&2
  exit 1
fi

mkdir -p "${TMUX_LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Run log: ${RUN_LOG}"
  exit 0
fi

cat >"${RUN_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT}"

export PROJECT_ROOT="${PROJECT_ROOT}"
export TOTAL_SAMPLES="${TOTAL_SAMPLES}"
export GPU_IDS="${GPU_IDS}"
export START_OFFSET="${START_OFFSET}"
export OVERWRITE="${OVERWRITE}"

echo "tmux session: ${SESSION_NAME}"
echo "project      : ${PROJECT_ROOT}"
echo "total_samples: ${TOTAL_SAMPLES}"
echo "start_offset : ${START_OFFSET}"
echo "gpu_ids      : ${GPU_IDS}"
echo "run_log      : ${RUN_LOG}"

bash changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb_multigpu.sh
EOF

chmod +x "${RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash '${RUN_SCRIPT}' 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Run log: ${RUN_LOG}"
