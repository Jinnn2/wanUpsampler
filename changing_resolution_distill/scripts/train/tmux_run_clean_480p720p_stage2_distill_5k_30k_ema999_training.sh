#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export GPU_IDS="${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CR_DISTILL_STAGE2_TAG="${CR_DISTILL_STAGE2_TAG:-14b_cfgdistill_5k}"
export MAX_STEPS="${MAX_STEPS:-30000}"
export EMA_DECAY="${EMA_DECAY:-0.999}"
export SESSION_NAME="${SESSION_NAME:-wan_cr_distill_stage2_clean_5k_30k_ema999_train}"
export WORKER_LOG_DIR="${WORKER_LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_distill_stage2_clean_5k_30k_ema999_train}"
export RUN_LOG="${RUN_LOG:-${PROJECT_ROOT}/logs/train_clean_480p720p_stage2_distill_5k_30k_ema999.log}"
export RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_ROOT}/logs/run_clean_480p720p_stage2_distill_5k_30k_ema999_train.tmux.sh}"

exec bash "${SCRIPT_DIR}/tmux_run_clean_480p720p_stage2_distill_lmdb_training.sh"
