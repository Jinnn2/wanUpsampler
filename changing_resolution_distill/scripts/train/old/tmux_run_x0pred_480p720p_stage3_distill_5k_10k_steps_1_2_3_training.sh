#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STEPS="${STEPS:-1,2,3}"
export GPU_IDS="${GPU_IDS:-0,1,2}"
export CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG:-14b_cfgdistill_5k}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SESSION_NAME="${SESSION_NAME:-wan_cr_distill_stage3_x0pred_5k_10k_steps_1_2_3_train}"

exec bash "${SCRIPT_DIR}/tmux_run_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3_training.sh"
