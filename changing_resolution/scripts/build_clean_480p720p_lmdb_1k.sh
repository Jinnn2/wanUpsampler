#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"

PROMPTS_DIR="${PROMPTS_DIR:-${PROJECT_ROOT}/prompts}"
PROMPTS_FILE="${CR_HF_PROMPTS_FILE:-${PROMPTS_DIR}/vidprom_filtered_extended.txt}"
RAW_VIDEO_DIR="${CR_RAW_VIDEO_DIR_1K:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p_1k}"
LMDB_DIR="${CR_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
START_SEED="${START_SEED:-520000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-16}"
PRECISION="${PRECISION:-bf16}"
SHARD_SIZE="${SHARD_SIZE:-100}"
MAP_SIZE_GB="${MAP_SIZE_GB:-256}"
OVERWRITE_LMDB="${OVERWRITE_LMDB:-0}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${LIGHTX2V_REPO}:${PYTHONPATH:-}"
export LIGHTX2V_REPO

check_paths() {
  if [[ ! -d "${LIGHTX2V_REPO}" ]]; then
    echo "LightX2V repo not found: ${LIGHTX2V_REPO}" >&2
    exit 1
  fi
  if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "Wan model root not found: ${MODEL_ROOT}" >&2
    exit 1
  fi
  if [[ ! -f "${VAE_PATH}" ]]; then
    echo "Wan VAE weights not found: ${VAE_PATH}" >&2
    exit 1
  fi
  python - <<'PY'
import importlib.util
missing = [name for name in ("lmdb", "torch", "imageio") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing python packages: " + ", ".join(missing) + ". Run: pip install -r requirements.txt")
PY
}

download_prompts() {
  mkdir -p "${PROMPTS_DIR}"
  if [[ -f "${PROMPTS_FILE}" ]]; then
    echo "Prompts already exist: ${PROMPTS_FILE}"
    return
  fi
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install huggingface_hub first: pip install huggingface_hub" >&2
    exit 1
  fi
  huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir "${PROMPTS_DIR}"
}

generate_videos() {
  local existing
  existing="$(find "${RAW_VIDEO_DIR}" -type f -name '*.mp4' 2>/dev/null | wc -l || true)"
  if (( existing >= NUM_SAMPLES )); then
    echo "Skip generation: ${RAW_VIDEO_DIR} already has ${existing} mp4 files"
    return
  fi

  CR_PROMPTS_FILE="${PROMPTS_FILE}" \
  CR_RAW_VIDEO_DIR="${RAW_VIDEO_DIR}" \
  MAX_PROMPTS="${NUM_SAMPLES}" \
  START_SEED="${START_SEED}" \
  bash "${PROJECT_ROOT}/changing_resolution/scripts/generate_wan21_720p_dataset.sh"
}

build_lmdb() {
  local overwrite_args=()
  if [[ "${OVERWRITE_LMDB}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi

  python "${PROJECT_ROOT}/changing_resolution/scripts/build_480p720p_lmdb.py" \
    --video_dir "${RAW_VIDEO_DIR}" \
    --out_dir "${LMDB_DIR}" \
    --prompts_file "${PROMPTS_FILE}" \
    --model_root "${MODEL_ROOT}" \
    --vae_path "${VAE_PATH}" \
    --wan_repo "${LIGHTX2V_REPO}" \
    --vae_backend lightx2v \
    --hr_size "${HR_H}" "${HR_W}" \
    --lr_size "${LR_H}" "${LR_W}" \
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

MODE="${1:-all}"
check_paths

case "${MODE}" in
  prompts)
    download_prompts
    ;;
  generate)
    download_prompts
    generate_videos
    ;;
  lmdb)
    download_prompts
    build_lmdb
    ;;
  all)
    download_prompts
    generate_videos
    build_lmdb
    ;;
  *)
    echo "Usage: bash changing_resolution/scripts/build_clean_480p720p_lmdb_1k.sh [prompts|generate|lmdb|all]" >&2
    exit 2
    ;;
esac

echo "1k clean-latent LMDB dataset path: ${LMDB_DIR}"
