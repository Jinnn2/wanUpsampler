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

RAW_VIDEO_DIR="${CR_RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p}"
LATENT_DIR="${CR_LATENT_DIR:-${PROJECT_ROOT}/data/changing_resolution/latent_pairs_480p720p}"
OUT_DIR="${CR_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p}"
CONFIG="${CR_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p.yaml}"
DATA_FORMAT="${DATA_FORMAT:-files}"

HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-16}"
PRECISION="${PRECISION:-bf16}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-100000}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-8}"
SCALE_FACTOR="${SCALE_FACTOR:-1.5}"
RESUME="${RESUME:-}"

MODE="${1:-all}"

export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
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
}

check_raw_videos() {
  if [[ ! -d "${RAW_VIDEO_DIR}" ]]; then
    echo "Raw video dir not found: ${RAW_VIDEO_DIR}" >&2
    echo "Run: bash changing_resolution/scripts/generate_wan21_720p_dataset.sh" >&2
    exit 1
  fi
  if [[ -z "$(find "${RAW_VIDEO_DIR}" -type f -name '*.mp4' -print -quit)" ]]; then
    echo "No mp4 videos found under: ${RAW_VIDEO_DIR}" >&2
    echo "Run: bash changing_resolution/scripts/generate_wan21_720p_dataset.sh" >&2
    exit 1
  fi
}

build_latents() {
  check_paths
  check_raw_videos
  python "${PROJECT_ROOT}/changing_resolution/scripts/build_480p720p_latents.py" \
    --video_dir "${RAW_VIDEO_DIR}" \
    --out_dir "${LATENT_DIR}" \
    --model_root "${MODEL_ROOT}" \
    --vae_path "${VAE_PATH}" \
    --wan_repo "${LIGHTX2V_REPO}" \
    --vae_backend lightx2v \
    --hr_size "${HR_H}" "${HR_W}" \
    --lr_size "${LR_H}" "${LR_W}" \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --precision "${PRECISION}" \
    --skip_bad_videos
}

train_model() {
  if [[ "${DATA_FORMAT}" == "files" ]]; then
    if [[ ! -d "${LATENT_DIR}" ]] || [[ -z "$(find "${LATENT_DIR}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit 2>/dev/null)" ]]; then
      echo "No latent samples found under: ${LATENT_DIR}" >&2
      echo "Run build first: bash changing_resolution/scripts/run_clean_480p720p_training.sh build" >&2
      exit 1
    fi
  elif [[ "${DATA_FORMAT}" == "lmdb" ]]; then
    if [[ ! -d "${LATENT_DIR}" ]] || [[ -z "$(find "${LATENT_DIR}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
      echo "No LMDB shards found under: ${LATENT_DIR}" >&2
      echo "Build LMDB first: bash changing_resolution/scripts/build_clean_480p720p_lmdb_1k.sh all" >&2
      exit 1
    fi
  else
    echo "DATA_FORMAT must be files or lmdb, got: ${DATA_FORMAT}" >&2
    exit 2
  fi

  local resume_args=()
  if [[ -n "${RESUME}" ]]; then
    resume_args=(--resume "${RESUME}")
  fi

  python "${PROJECT_ROOT}/changing_resolution/scripts/train_clean_latent_resizer.py" \
    --config "${CONFIG}" \
    --data_dir "${LATENT_DIR}" \
    --data_format "${DATA_FORMAT}" \
    --out_dir "${OUT_DIR}" \
    --hidden_channels "${HIDDEN_CHANNELS}" \
    --num_res_blocks "${NUM_RES_BLOCKS}" \
    --scale_factor "${SCALE_FACTOR}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max_steps "${MAX_STEPS}" \
    --precision "${PRECISION}" \
    "${resume_args[@]}"
}

case "${MODE}" in
  generate)
    bash "${PROJECT_ROOT}/changing_resolution/scripts/generate_wan21_720p_dataset.sh"
    ;;
  build)
    build_latents
    ;;
  train)
    train_model
    ;;
  all)
    bash "${PROJECT_ROOT}/changing_resolution/scripts/generate_wan21_720p_dataset.sh"
    build_latents
    train_model
    ;;
  *)
    echo "Usage: bash changing_resolution/scripts/run_clean_480p720p_training.sh [generate|build|train|all]" >&2
    exit 2
    ;;
esac
