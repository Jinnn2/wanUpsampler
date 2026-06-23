#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export LORA_RANK="${LORA_RANK:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
export TRAINING_MODE="${TRAINING_MODE:-on_policy}"
export ON_POLICY_LOSS_TYPE="${ON_POLICY_LOSS_TYPE:-velocity_target}"
export ON_POLICY_ACTIVE_STEPS="${ON_POLICY_ACTIVE_STEPS:-all_before_train}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-4e-5}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export CR_DISTILL_TEACHER_TRAJ_LMDB_DIR="${CR_DISTILL_TEACHER_TRAJ_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_teacher_trajectory_lora_14b_cfgdistill_5k_step3}"
export CR_DISTILL_TEACHER_TRAJ_OUT_DIR="${CR_DISTILL_TEACHER_TRAJ_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_teacher_trajectory_lora_plan_e_on_policy_velocity_rank16_qkvo_ffn}"

bash "${SCRIPT_DIR}/run_teacher_trajectory_lora_training.sh" "${1:-train}"
