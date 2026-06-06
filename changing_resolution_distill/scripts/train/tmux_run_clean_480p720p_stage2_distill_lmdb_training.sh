#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-wan_cr_distill_stage2_clean_train}"

GPU_IDS="${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}}"
GPU_ID="${GPU_ID:-${GPU_IDS%%,*}}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
CR_DISTILL_STAGE2_TAG="${CR_DISTILL_STAGE2_TAG:-14b_cfgdistill_5k}"
MAX_STEPS="${MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
EMA_DECAY="${EMA_DECAY:-}"
PRECISION="${PRECISION:-bf16}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-8}"
SCALE_FACTOR="${SCALE_FACTOR:-1.5}"
NO_RESIDUAL_SKIP="${NO_RESIDUAL_SKIP:-true}"
AUTO_RESUME="${AUTO_RESUME:-1}"

TMUX_LOG_DIR="${TMUX_LOG_DIR:-${PROJECT_ROOT}/logs}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_distill_stage2_clean_train}"
RUN_LOG="${RUN_LOG:-${TMUX_LOG_DIR}/train_clean_480p720p_stage2_distill.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${TMUX_LOG_DIR}/run_clean_480p720p_stage2_distill_train.tmux.sh}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run the Stage 2 distill training script directly." >&2
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
export GPU_ID="${GPU_ID}"
export LIGHTX2V_REPO="${LIGHTX2V_REPO}"
export CR_DISTILL_STAGE2_TAG="${CR_DISTILL_STAGE2_TAG}"
export MAX_STEPS="${MAX_STEPS}"
export BATCH_SIZE="${BATCH_SIZE}"
export GRAD_ACCUM="${GRAD_ACCUM}"
export LR="${LR}"
export EMA_DECAY="${EMA_DECAY}"
export PRECISION="${PRECISION}"
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS}"
export NUM_RES_BLOCKS="${NUM_RES_BLOCKS}"
export SCALE_FACTOR="${SCALE_FACTOR}"
export NO_RESIDUAL_SKIP="${NO_RESIDUAL_SKIP}"
export AUTO_RESUME="${AUTO_RESUME}"
export WORKER_LOG_DIR="${WORKER_LOG_DIR}"

out_dir="${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_480p720p_stage2_${CR_DISTILL_STAGE2_TAG}_lmdb"
resume_path=""
if [[ "${AUTO_RESUME}" == "1" && -f "\${out_dir}/latest.pt" ]]; then
  resume_path="\${out_dir}/latest.pt"
fi

worker_log="${WORKER_LOG_DIR}/stage2_clean_gpu_${GPU_ID}.log"

echo "tmux session: ${SESSION_NAME}"
echo "project     : ${PROJECT_ROOT}"
echo "gpu_ids     : ${GPU_IDS}"
echo "gpu_id      : ${GPU_ID}"
echo "max_steps   : ${MAX_STEPS}"
echo "ema_decay   : ${EMA_DECAY:-config default}"
echo "stage2_tag  : ${CR_DISTILL_STAGE2_TAG}"
echo "auto_resume : ${AUTO_RESUME}"
echo "run_log     : ${RUN_LOG}"
echo "worker_log  : \${worker_log}"
if [[ -n "\${resume_path}" ]]; then
  echo "resume      : \${resume_path}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
CR_DISTILL_STAGE2_LMDB_DIR="${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_clean_480p720p_${CR_DISTILL_STAGE2_TAG}" \
CR_DISTILL_STAGE2_OUT_DIR="\${out_dir}" \
RESUME="\${resume_path}" \
bash changing_resolution_distill/scripts/train/run_clean_480p720p_stage2_distill_lmdb_training.sh train \
  >"\${worker_log}" 2>&1

echo "Stage 2 distill clean-latent training finished."
echo "Worker log: \${worker_log}"
EOF

chmod +x "${RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash '${RUN_SCRIPT}' 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Run log: ${RUN_LOG}"
echo "Worker log: ${WORKER_LOG_DIR}/stage2_clean_gpu_${GPU_ID}.log"
