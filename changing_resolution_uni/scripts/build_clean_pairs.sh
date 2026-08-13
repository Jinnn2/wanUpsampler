#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VIDEO_DIR="${VIDEO_DIR:?Set VIDEO_DIR to generated HR videos}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
WAN_REPO="${WAN_REPO:-${LIGHTX2V_REPO:-}}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
SCALES="${SCALES:-1.5 2.0 3.0}"
LR_SIZES="${LR_SIZES:-}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
VAE_BACKEND="${VAE_BACKEND:-auto}"
RESIZE_MODE="${RESIZE_MODE:-bicubic}"
MAP_SIZE_GB="${MAP_SIZE_GB:-256}"

cd "${PROJECT_ROOT}"
args=(
  -m changing_resolution_uni.build_latent_pairs
  --video_dir "${VIDEO_DIR}"
  --out_dir "${OUT_DIR}"
  --model_root "${MODEL_ROOT}"
  --vae_path "${VAE_PATH}"
  --hr_size "${HR_H}" "${HR_W}"
  --scales ${SCALES}
  --num_frames "${NUM_FRAMES}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --vae_backend "${VAE_BACKEND}"
  --resize_mode "${RESIZE_MODE}"
  --map_size_gb "${MAP_SIZE_GB}"
)
if [[ -n "${LR_SIZES}" ]]; then args+=(--lr_sizes ${LR_SIZES}); fi
if [[ -n "${WAN_REPO}" ]]; then args+=(--wan_repo "${WAN_REPO}"); fi
python "${args[@]}"
