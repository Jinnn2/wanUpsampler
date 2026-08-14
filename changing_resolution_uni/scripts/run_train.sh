#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/changing_resolution_uni/configs/train_universal_clean.yaml}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_uni_clean}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
RESUME="${RESUME:-}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

launcher=(python)
if (( NUM_GPUS > 1 )); then
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
fi

args=("${PROJECT_ROOT}/changing_resolution_uni/train.py" --config "${CONFIG}" --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" --max_steps "${MAX_STEPS}")
if [[ -n "${RESUME}" ]]; then args+=(--resume "${RESUME}"); fi
"${launcher[@]}" "${args[@]}"
