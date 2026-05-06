#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_lmdb_480p720p_1k_multigpu}"
MODE="${MODE:-all}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
START_SEED="${START_SEED:-520000}"
OVERWRITE_LMDB="${OVERWRITE_LMDB:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-8}"

TMUX_LOG_DIR="${TMUX_LOG_DIR:-${PROJECT_ROOT}/logs}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_lmdb_1k_multigpu}"
RUN_LOG="${RUN_LOG:-${TMUX_LOG_DIR}/build_clean_lmdb_480p720p_1k_multigpu.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${TMUX_LOG_DIR}/run_clean_lmdb_480p720p_1k_multigpu.tmux.sh}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run build_clean_480p720p_lmdb_1k_multigpu.sh directly." >&2
  exit 1
fi

mkdir -p "${TMUX_LOG_DIR}" "${WORKER_LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Run log: ${RUN_LOG}"
  echo "Worker logs: ${WORKER_LOG_DIR}/part_*.log"
  exit 0
fi

cat >"${RUN_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT}"

export PROJECT_ROOT="${PROJECT_ROOT}"
export TOTAL_SAMPLES="${TOTAL_SAMPLES}"
export GPU_IDS="${GPU_IDS}"
export START_SEED="${START_SEED}"
export OVERWRITE_LMDB="${OVERWRITE_LMDB}"
export MONITOR_INTERVAL="${MONITOR_INTERVAL}"
export MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES}"
export LOG_DIR="${WORKER_LOG_DIR}"

echo "tmux session: ${SESSION_NAME}"
echo "project      : ${PROJECT_ROOT}"
echo "mode         : ${MODE}"
echo "total_samples: ${TOTAL_SAMPLES}"
echo "gpu_ids      : ${GPU_IDS}"
echo "run_log      : ${RUN_LOG}"
echo "worker_logs  : ${WORKER_LOG_DIR}/part_*.log"

bash changing_resolution/scripts/build_clean_480p720p_lmdb_1k_multigpu.sh "${MODE}"
EOF

chmod +x "${RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash '${RUN_SCRIPT}' 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Run log: ${RUN_LOG}"
echo "Worker logs: ${WORKER_LOG_DIR}/part_*.log"
