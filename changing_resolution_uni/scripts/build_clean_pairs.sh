#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DEFAULT_LIGHTX2V_REPO="/mnt/afs_2/houze/LightX2V"
if [[ -d "${PROJECT_ROOT}/../LightX2V" ]]; then
  DEFAULT_LIGHTX2V_REPO="$(cd "${PROJECT_ROOT}/../LightX2V" && pwd)"
fi
VIDEO_DIR="${VIDEO_DIR:-${CR_RAW_VIDEO_DIR_1K:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p_1k}}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
WAN_REPO="${WAN_REPO:-${LIGHTX2V_REPO:-${DEFAULT_LIGHTX2V_REPO}}}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
SCALES="${SCALES:-1.5 2.0 3.0}"
LR_SIZES="${LR_SIZES:-480x832 368x640 240x416}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
VAE_BACKEND="${VAE_BACKEND:-auto}"
RESIZE_MODE="${RESIZE_MODE:-bicubic}"
MAP_SIZE_GB="${MAP_SIZE_GB:-256}"

for path in "${VIDEO_DIR}" "${MODEL_ROOT}" "${WAN_REPO}"; do
  [[ -d "${path}" ]] || { echo "Required directory not found: ${path}" >&2; exit 2; }
done
[[ -f "${VAE_PATH}" ]] || { echo "Wan VAE weights not found: ${VAE_PATH}" >&2; exit 2; }

export LIGHTX2V_REPO="${WAN_REPO}"
export PYTHONPATH="${WAN_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
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
