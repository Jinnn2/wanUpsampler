#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export LORA_RANK="${LORA_RANK:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MAX_STEPS="${MAX_STEPS:-4000}"
export CR_DISTILL_LORA_OUT_DIR="${CR_DISTILL_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_plan_c_rank16_qkvo}"

bash "${SCRIPT_DIR}/run_last_step_skip_lora_training.sh" "${1:-train}"
