#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
CALLER_NUM_GPUS="${NUM_GPUS:-}"
CALLER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

export PROJECT_ROOT
export CR_LMDB_DIR="${CR_LMDB_368X640_720X1248_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_368x640_720x1248_1k}"
export CR_STAGE2_CONFIG="${CR_STAGE2_368X640_720X1248_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml}"
export CR_STAGE2_OUT_DIR="${CR_STAGE2_368X640_720X1248_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb}"
export SCALE_FACTOR="2.0"
export NUM_GPUS="${CALLER_NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CALLER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Keep the original single-GPU effective batch: 1 batch * 8 accumulation = 8.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

if [[ "${1:-train}" == "model_check" ]]; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python - <<'PY'
import torch
from wan_sr.models import WanCleanLatentResizerStage2

model = WanCleanLatentResizerStage2(
    hidden_channels=32,
    num_res_blocks=2,
    scale_factor=2.0,
    resize_op="conv3d_pixel_shuffle_crop",
)
x = torch.randn(1, 16, 2, 4, 6)
y = model(x, output_size=(6, 10))
expected = (1, 16, 2, 6, 10)
if tuple(y.shape) != expected:
    raise SystemExit(f"Unexpected Stage2 output: {tuple(y.shape)} != {expected}")
print(f"368p Stage2 model check passed: {tuple(x.shape)} -> {tuple(y.shape)}")
PY
  exit 0
fi

exec bash "${PROJECT_ROOT}/changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh" "${1:-train}"
