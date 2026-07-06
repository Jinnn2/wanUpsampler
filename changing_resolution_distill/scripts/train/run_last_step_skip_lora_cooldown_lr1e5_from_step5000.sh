#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

SOURCE_LORA_CKPT="${COOLDOWN_SOURCE_LORA_CKPT:-${SOURCE_LORA_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/step_0005000.safetensors}}"
COOLDOWN_OUT_DIR="${COOLDOWN_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3_cooldown_lr1e5_from_step5000}"

export SOURCE_LORA_CKPT
export CR_DISTILL_LORA_OUT_DIR="${COOLDOWN_OUT_DIR}"
export LORA_CHECKPOINT="${SOURCE_LORA_CKPT}"
export LR="${COOLDOWN_LR:-1e-5}"
export MAX_STEPS="${COOLDOWN_MAX_STEPS:-2000}"
export RESUME=""

echo "Starting fresh-optimizer LoRA cooldown from:"
echo "  ${LORA_CHECKPOINT}"
echo "Output:"
echo "  ${CR_DISTILL_LORA_OUT_DIR}"
echo "LR=${LR} MAX_STEPS=${MAX_STEPS}"

exec bash "${SCRIPT_DIR}/run_last_step_skip_lora_training.sh" train
