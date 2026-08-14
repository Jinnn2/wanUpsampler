#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

CONFIG="${CONFIG:-${PROJECT_ROOT}/changing_resolution_uni/configs/train_universal_clean.yaml}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean_v1_1k}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_uni_clean_v1_1k}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-50000}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
RUN_LOG="${RUN_LOG:-${OUT_DIR}/train.log}"

[[ -f "${CONFIG}" ]] || { echo "Training config not found: ${CONFIG}" >&2; exit 2; }
[[ -d "${DATA_DIR}" ]] || { echo "Training data directory not found: ${DATA_DIR}" >&2; exit 2; }

shopt -s nullglob
shards=("${DATA_DIR}"/shard_*)
shopt -u nullglob
if (( ${#shards[@]} == 0 )); then
  echo "No shard_* directories found under ${DATA_DIR}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
NUM_GPUS="${#gpu_array[@]}"
if (( NUM_GPUS < 1 )); then
  echo "GPU_IDS must contain at least one device id" >&2
  exit 2
fi

if [[ -z "${RESUME}" && "${AUTO_RESUME}" == "1" && -f "${OUT_DIR}/last.pt" ]]; then
  RESUME="${OUT_DIR}/last.pt"
fi
if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
  echo "Resume checkpoint not found: ${RESUME}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "$(dirname "${RUN_LOG}")"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launcher=(python)
if (( NUM_GPUS > 1 )); then
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
fi

args=(
  "${PROJECT_ROOT}/changing_resolution_uni/train.py"
  --config "${CONFIG}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --max_steps "${MAX_STEPS}"
)
if [[ -n "${RESUME}" ]]; then
  args+=(--resume "${RESUME}")
fi

echo "U-ITU formal training"
echo "  project     : ${PROJECT_ROOT}"
echo "  config      : ${CONFIG}"
echo "  data        : ${DATA_DIR} (${#shards[@]} shards)"
echo "  output      : ${OUT_DIR}"
echo "  GPUs        : ${GPU_IDS} (${NUM_GPUS})"
echo "  max steps   : ${MAX_STEPS}"
echo "  resume      : ${RESUME:-none}"
echo "  log         : ${RUN_LOG}"

cd "${PROJECT_ROOT}"
"${launcher[@]}" "${args[@]}" 2>&1 | tee -a "${RUN_LOG}"
