#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DENOISE_STEP="${DENOISE_STEP:-45}"
SESSION_NAME="${SESSION_NAME:-wan_cr_stage3_x0pred_lmdb_step${DENOISE_STEP}_multigpu}"
USER_CR_STAGE3_LMDB_DIR="${CR_STAGE3_LMDB_DIR+x}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
START_OFFSET="${START_OFFSET:-0}"
OVERWRITE="${OVERWRITE:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-8}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/path/to/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/Wan-AI/Wan2.1-T2V-1.3B}"
CR_STAGE2_LMDB_DIR="${CR_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CR_STAGE3_X0PRED_CONFIG="${CR_STAGE3_X0PRED_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_x0pred_480p.json}"
INFER_STEPS="${INFER_STEPS:-50}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
HR_TARGET_MODE="${HR_TARGET_MODE:-x0_pred}"
HR_SEED_OFFSET="${HR_SEED_OFFSET:-0}"
case "${HR_TARGET_MODE}" in
  x0_pred) DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_x0predhr_step${DENOISE_STEP}" ;;
  clean) DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_cleanhr_step${DENOISE_STEP}" ;;
  *) DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_${HR_TARGET_MODE}_step${DENOISE_STEP}" ;;
esac
BASE_SEED="${BASE_SEED:-9300}"
MODE="${MODE:-lightx2v}"
PRECISION="${PRECISION:-bf16}"
if [[ -z "${USER_CR_STAGE3_LMDB_DIR}" ]]; then
  CR_STAGE3_LMDB_DIR="${PROJECT_ROOT}/data/changing_resolution/${DEFAULT_LMDB_NAME}"
fi

TMUX_LOG_DIR="${TMUX_LOG_DIR:-${PROJECT_ROOT}/logs}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_stage3_x0pred_lmdb_step${DENOISE_STEP}_multigpu}"
RUN_LOG="${RUN_LOG:-${TMUX_LOG_DIR}/build_x0pred_480p720p_stage3_lmdb_step${DENOISE_STEP}_multigpu.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${TMUX_LOG_DIR}/run_x0pred_480p720p_stage3_lmdb_step${DENOISE_STEP}_multigpu.tmux.sh}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run build_x0pred_480p720p_stage3_lmdb_multigpu.sh directly." >&2
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
export START_OFFSET="${START_OFFSET}"
export OVERWRITE="${OVERWRITE}"
export MONITOR_INTERVAL="${MONITOR_INTERVAL}"
export MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES}"
export LOG_DIR="${WORKER_LOG_DIR}"
export LIGHTX2V_REPO="${LIGHTX2V_REPO}"
export MODEL_ROOT="${MODEL_ROOT}"
export CR_STAGE2_LMDB_DIR="${CR_STAGE2_LMDB_DIR}"
export CR_STAGE3_LMDB_DIR="${CR_STAGE3_LMDB_DIR}"
export CR_STAGE3_X0PRED_CONFIG="${CR_STAGE3_X0PRED_CONFIG}"
export INFER_STEPS="${INFER_STEPS}"
export DENOISE_STEP="${DENOISE_STEP}"
export SAMPLE_SHIFT="${SAMPLE_SHIFT}"
export GUIDE_SCALE="${GUIDE_SCALE}"
export HR_TARGET_MODE="${HR_TARGET_MODE}"
export HR_SEED_OFFSET="${HR_SEED_OFFSET}"
export BASE_SEED="${BASE_SEED}"
export MODE="${MODE}"
export PRECISION="${PRECISION}"

echo "tmux session: ${SESSION_NAME}"
echo "project      : ${PROJECT_ROOT}"
echo "total_samples: ${TOTAL_SAMPLES}"
echo "start_offset : ${START_OFFSET}"
echo "gpu_ids      : ${GPU_IDS}"
echo "denoise_step : ${DENOISE_STEP}"
echo "hr_target    : ${HR_TARGET_MODE}"
echo "mode         : ${MODE}"
echo "run_log      : ${RUN_LOG}"
echo "worker_logs  : ${WORKER_LOG_DIR}/part_*.log"

bash changing_resolution/scripts/data/build_x0pred_480p720p_stage3_lmdb_multigpu.sh
EOF

chmod +x "${RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash '${RUN_SCRIPT}' 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Run log: ${RUN_LOG}"
echo "Worker logs: ${WORKER_LOG_DIR}/part_*.log"
