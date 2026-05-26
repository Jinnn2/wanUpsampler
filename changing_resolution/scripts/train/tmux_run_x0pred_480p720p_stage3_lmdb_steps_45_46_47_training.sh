#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
STEPS="${STEPS:-45,46,47}"
STEP_TAG="${STEPS//,/_}"
SESSION_NAME="${SESSION_NAME:-wan_cr_stage3_x0pred_lmdb_steps_${STEP_TAG}_train}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
PRECISION="${PRECISION:-bf16}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-8}"
SCALE_FACTOR="${SCALE_FACTOR:-1.5}"
NO_RESIDUAL_SKIP="${NO_RESIDUAL_SKIP:-true}"

TMUX_LOG_DIR="${TMUX_LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${TMUX_LOG_DIR}/train_x0pred_480p720p_stage3_lmdb_steps_${STEP_TAG}.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${TMUX_LOG_DIR}/run_x0pred_480p720p_stage3_lmdb_steps_${STEP_TAG}_train.tmux.sh}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run the per-step training scripts directly." >&2
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export LIGHTX2V_REPO="${LIGHTX2V_REPO}"
export MAX_STEPS="${MAX_STEPS}"
export BATCH_SIZE="${BATCH_SIZE}"
export GRAD_ACCUM="${GRAD_ACCUM}"
export LR="${LR}"
export PRECISION="${PRECISION}"
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS}"
export NUM_RES_BLOCKS="${NUM_RES_BLOCKS}"
export SCALE_FACTOR="${SCALE_FACTOR}"
export NO_RESIDUAL_SKIP="${NO_RESIDUAL_SKIP}"

echo "tmux session: ${SESSION_NAME}"
echo "project     : ${PROJECT_ROOT}"
echo "steps       : ${STEPS}"
echo "gpu         : ${CUDA_VISIBLE_DEVICES}"
echo "max_steps   : ${MAX_STEPS}"
echo "run_log     : ${RUN_LOG}"

IFS=',' read -r -a STEP_LIST <<< "${STEPS}"
for step in "\${STEP_LIST[@]}"; do
  step="\$(echo "\${step}" | xargs)"
  if [[ -z "\${step}" ]]; then
    continue
  fi
  export DENOISE_STEP="\${step}"
  export CR_STAGE3_LMDB_DIR="${PROJECT_ROOT}/data/changing_resolution/lmdb_x0pred_480p720p_stage3_step\${step}"
  export CR_STAGE3_OUT_DIR="${PROJECT_ROOT}/outputs/changing_resolution_x0pred_480p720p_stage3_step\${step}_lmdb"

  echo "===== train Stage 3 x0-pred resizer step \${step} ====="
  bash changing_resolution/scripts/train/run_x0pred_480p720p_stage3_lmdb_training.sh train
done

echo "All Stage 3 x0-pred trainings finished for steps: ${STEPS}"
EOF

chmod +x "${RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash '${RUN_SCRIPT}' 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Run log: ${RUN_LOG}"
