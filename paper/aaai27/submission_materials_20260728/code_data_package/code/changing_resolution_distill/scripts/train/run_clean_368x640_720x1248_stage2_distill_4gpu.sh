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

export PROJECT_ROOT
export CR_LMDB_DIR="${CR_DISTILL_360_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_clean_368x640_720x1248_14b_cfgdistill_5k}"
export CR_STAGE2_CONFIG="${CR_DISTILL_360_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml}"
export CR_STAGE2_OUT_DIR="${CR_DISTILL_360_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb}"
export SCALE_FACTOR=2.0
export CUDA_VISIBLE_DEVICES="${CALLER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

AVAILABLE_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if (( AVAILABLE_GPUS < 1 )); then
  echo "No CUDA device is available; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi
export NUM_GPUS="${NUM_GPUS:-${AVAILABLE_GPUS}}"
if (( NUM_GPUS > AVAILABLE_GPUS )); then
  echo "NUM_GPUS=${NUM_GPUS} exceeds the ${AVAILABLE_GPUS} CUDA device(s) available at runtime." >&2
  exit 2
fi

# Keep the historical effective batch and total loader-worker budget close to 8
# whether this job runs on one, two, or four allocated GPUs.
export GRAD_ACCUM="${GRAD_ACCUM:-$(( (8 + NUM_GPUS - 1) / NUM_GPUS ))}"
export NUM_WORKERS="${NUM_WORKERS:-$(( (8 + NUM_GPUS - 1) / NUM_GPUS ))}"
echo "Distill Stage2 runtime: available_gpus=${AVAILABLE_GPUS} world_size=${NUM_GPUS} grad_accum=${GRAD_ACCUM}"

MODE="${1:-train}"
if [[ "${MODE}" == "model_check" ]]; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python - <<'PY'
import torch
from wan_sr.models import WanCleanLatentResizerStage2

model = WanCleanLatentResizerStage2(
    hidden_channels=32,
    num_res_blocks=2,
    scale_factor=2.0,
    resize_op="conv3d_pixel_shuffle_crop",
)
x = torch.randn(1, 16, 2, 46, 80)
y = model(x, output_size=(90, 156))
expected = (1, 16, 2, 90, 156)
if tuple(y.shape) != expected:
    raise SystemExit(f"Unexpected Stage2 output: {tuple(y.shape)} != {expected}")
print(f"Distill 360p Stage2 model check passed: {tuple(x.shape)} -> {tuple(y.shape)}")
PY
  exit 0
fi

if [[ "${AUTO_RESUME:-1}" == "1" && -z "${RESUME:-}" && -f "${CR_STAGE2_OUT_DIR}/latest.pt" ]]; then
  export RESUME="${CR_STAGE2_OUT_DIR}/latest.pt"
  echo "Auto-resume: ${RESUME}"
fi

exec bash "${PROJECT_ROOT}/changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh" "${MODE}"
