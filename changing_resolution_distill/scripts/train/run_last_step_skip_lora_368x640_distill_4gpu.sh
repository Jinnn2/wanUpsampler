#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
CALLER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

export CR_DISTILL_LORA_CONFIG="${CR_DISTILL_360_LORA_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_last_step_skip_lora_368x640_distill.yaml}"
export CR_DISTILL_LORA_LMDB_DIR="${CR_DISTILL_360_LORA_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3}"
export CR_DISTILL_LORA_OUT_DIR="${CR_DISTILL_360_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3}"
export CUDA_VISIBLE_DEVICES="${CALLER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS=4
export BASE_GRAD_ACCUM=8

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi

exec bash "${PROJECT_ROOT}/changing_resolution_distill/scripts/train/run_last_step_skip_lora_training.sh" "${1:-train}"
