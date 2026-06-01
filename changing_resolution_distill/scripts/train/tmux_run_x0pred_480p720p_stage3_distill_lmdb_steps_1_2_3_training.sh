#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
STEPS="${STEPS:-1,2,3}"
STEP_TAG="${STEPS//,/_}"
SESSION_NAME="${SESSION_NAME:-wan_cr_distill_stage3_x0pred_steps_${STEP_TAG}_train}"

GPU_IDS="${GPU_IDS:-0,1,2}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG:-14b_cfgdistill}"
MAX_STEPS="${MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
PRECISION="${PRECISION:-bf16}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-8}"
SCALE_FACTOR="${SCALE_FACTOR:-1.5}"
NO_RESIDUAL_SKIP="${NO_RESIDUAL_SKIP:-true}"

TMUX_LOG_DIR="${TMUX_LOG_DIR:-${PROJECT_ROOT}/logs}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_distill_stage3_x0pred_steps_${STEP_TAG}_train}"
RUN_LOG="${RUN_LOG:-${TMUX_LOG_DIR}/train_x0pred_480p720p_stage3_distill_steps_${STEP_TAG}.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${TMUX_LOG_DIR}/run_x0pred_480p720p_stage3_distill_steps_${STEP_TAG}_train.tmux.sh}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run the per-step training scripts directly." >&2
  exit 1
fi

mkdir -p "${TMUX_LOG_DIR}" "${WORKER_LOG_DIR}"

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
export GPU_IDS="${GPU_IDS}"
export LIGHTX2V_REPO="${LIGHTX2V_REPO}"
export CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG}"
export MAX_STEPS="${MAX_STEPS}"
export BATCH_SIZE="${BATCH_SIZE}"
export GRAD_ACCUM="${GRAD_ACCUM}"
export LR="${LR}"
export PRECISION="${PRECISION}"
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS}"
export NUM_RES_BLOCKS="${NUM_RES_BLOCKS}"
export SCALE_FACTOR="${SCALE_FACTOR}"
export NO_RESIDUAL_SKIP="${NO_RESIDUAL_SKIP}"
export WORKER_LOG_DIR="${WORKER_LOG_DIR}"

echo "tmux session: ${SESSION_NAME}"
echo "project     : ${PROJECT_ROOT}"
echo "steps       : ${STEPS}"
echo "gpu_ids     : ${GPU_IDS}"
echo "max_steps   : ${MAX_STEPS}"
echo "stage3_tag  : ${CR_DISTILL_STAGE3_TAG}"
echo "run_log     : ${RUN_LOG}"
echo "worker_logs : ${WORKER_LOG_DIR}/step_*.log"

IFS=',' read -r -a STEP_LIST <<< "${STEPS}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"

if (( \${#GPU_LIST[@]} < \${#STEP_LIST[@]} )); then
  echo "Need at least one GPU per step: steps=${STEPS}, gpu_ids=${GPU_IDS}" >&2
  exit 2
fi

pids=()
worker_names=()
worker_logs=()

cleanup_workers() {
  if (( \${#pids[@]} > 0 )); then
    echo "Stopping training workers..." >&2
    kill "\${pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_workers; exit 130' INT TERM

for index in "\${!STEP_LIST[@]}"; do
  step="\$(echo "\${STEP_LIST[\${index}]}" | xargs)"
  if [[ -z "\${step}" ]]; then
    continue
  fi
  gpu="\$(echo "\${GPU_LIST[\${index}]}" | xargs)"
  if [[ -z "\${gpu}" ]]; then
    echo "Empty GPU id for step \${step}" >&2
    exit 2
  fi

  log_path="${WORKER_LOG_DIR}/step_\${step}_gpu_\${gpu}.log"
  echo "Launch distill step \${step}: gpu=\${gpu}, log=\${log_path}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="\${gpu}" \
    HANDOFF_STEP="\${step}" \
    CR_DISTILL_STAGE3_LMDB_DIR="${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step\${step}" \
    CR_DISTILL_STAGE3_OUT_DIR="${PROJECT_ROOT}/outputs/changing_resolution_distill_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step\${step}_lmdb" \
    bash changing_resolution_distill/scripts/train/run_x0pred_480p720p_stage3_distill_lmdb_training.sh train
  ) >"\${log_path}" 2>&1 &

  pids+=("\$!")
  worker_names+=("step_\${step}_gpu_\${gpu}")
  worker_logs+=("\${log_path}")
done

failed=0
for index in "\${!pids[@]}"; do
  pid="\${pids[\${index}]}"
  name="\${worker_names[\${index}]}"
  log_path="\${worker_logs[\${index}]}"
  if wait "\${pid}"; then
    echo "[done] \${name} log=\${log_path}"
  else
    echo "[failed] \${name}. Last log lines:"
    if [[ -f "\${log_path}" ]]; then
      tail -n 120 "\${log_path}" || true
    fi
    failed=1
  fi
done

trap - INT TERM

if (( failed != 0 )); then
  echo "At least one Stage 3 distill training worker failed. Check logs under: ${WORKER_LOG_DIR}" >&2
  exit 1
fi

echo "All Stage 3 distill x0-pred trainings finished for steps: ${STEPS}"
EOF

chmod +x "${RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash '${RUN_SCRIPT}' 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Run log: ${RUN_LOG}"
echo "Worker logs: ${WORKER_LOG_DIR}/step_*.log"
