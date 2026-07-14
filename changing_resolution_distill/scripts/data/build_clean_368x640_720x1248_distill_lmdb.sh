#!/usr/bin/env bash
set -euo pipefail

# Build a dedicated distill clean-latent dataset from existing 720x1248
# CfgDistill videos. This never writes to the 480p Stage2 LMDB.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
VAE_PATH="${CR_DISTILL_VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
PROMPTS_FILE="${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}"
RAW_VIDEO_DIR="${CR_DISTILL_RAW_VIDEO_DIR_1K:-${PROJECT_ROOT}/data/changing_resolution_distill/raw_wan21_14b_cfgdistill_720p_5k}"
LMDB_DIR="${CR_DISTILL_360_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_clean_368x640_720x1248_14b_cfgdistill_5k}"

NUM_SAMPLES="${NUM_SAMPLES:-5000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-16}"
PRECISION="${PRECISION:-bf16}"
SHARD_SIZE="${SHARD_SIZE:-100}"
MAP_SIZE_GB="${MAP_SIZE_GB:-256}"
OVERWRITE_LMDB="${OVERWRITE_LMDB:-0}"

export CUDA_VISIBLE_DEVICES LIGHTX2V_REPO
export PYTHONPATH="${PROJECT_ROOT}:${LIGHTX2V_REPO}:${PYTHONPATH:-}"

check_paths() {
  for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}" "${RAW_VIDEO_DIR}"; do
    [[ -d "${path}" ]] || { echo "Directory not found: ${path}" >&2; exit 1; }
  done
  for path in "${VAE_PATH}" "${PROMPTS_FILE}"; do
    [[ -f "${path}" ]] || { echo "File not found: ${path}" >&2; exit 1; }
  done
  python - <<'PY'
import importlib.util
missing = [name for name in ("lmdb", "torch", "imageio") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing python packages: " + ", ".join(missing))
PY
}

build_lmdb() {
  local overwrite_args=()
  if [[ "${OVERWRITE_LMDB}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/data/build_clean_368x640_720x1248_distill_lmdb.py" \
    --video_dir "${RAW_VIDEO_DIR}" \
    --out_dir "${LMDB_DIR}" \
    --prompts_file "${PROMPTS_FILE}" \
    --model_root "${MODEL_ROOT}" \
    --vae_path "${VAE_PATH}" \
    --wan_repo "${LIGHTX2V_REPO}" \
    --vae_backend lightx2v \
    --hr_size 720 1248 \
    --lr_size 368 640 \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --max_samples "${NUM_SAMPLES}" \
    --require_samples "${NUM_SAMPLES}" \
    --shard_size "${SHARD_SIZE}" \
    --map_size_gb "${MAP_SIZE_GB}" \
    --precision "${PRECISION}" \
    --skip_bad_videos \
    "${overwrite_args[@]}"
}

MODE="${1:-lmdb}"
check_paths
case "${MODE}" in
  check)
    echo "Distill 360p LMDB preflight passed: ${RAW_VIDEO_DIR} -> ${LMDB_DIR}"
    ;;
  lmdb)
    build_lmdb
    echo "Distill 360p clean-latent LMDB ready: ${LMDB_DIR}"
    ;;
  *)
    echo "Usage: bash changing_resolution_distill/scripts/data/build_clean_368x640_720x1248_distill_lmdb.sh [check|lmdb]" >&2
    exit 2
    ;;
esac
