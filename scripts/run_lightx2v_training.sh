#!/usr/bin/env bash
set -euo pipefail

# LightX2V + Wan2.1 default paths on the current training machine.
# Override any variable from the shell when tuning, for example:
#   MAX_STEPS=20000 LR=5e-5 bash scripts/run_lightx2v_training.sh train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PATH_CONFIG="${SCRIPT_DIR}/../configs/local_paths.sh"
PATH_CONFIG="${PATH_CONFIG:-${DEFAULT_PATH_CONFIG}}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/data/yongyang/Jin/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"

RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/raw_videos}"
LATENT_DIR="${LATENT_DIR:-${PROJECT_ROOT}/data/latent_pairs_wan21_512}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/wan_traj_upsampler_x2}"
CONFIG="${CONFIG:-${TRAIN_CONFIG:-${PROJECT_ROOT}/configs/train_wan21_x2_512.yaml}}"

HR_H="${HR_H:-512}"
HR_W="${HR_W:-512}"
LR_H="${LR_H:-256}"
LR_W="${LR_W:-256}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-16}"
PRECISION="${PRECISION:-bf16}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-100000}"
WARMUP_CLEAN_STEPS="${WARMUP_CLEAN_STEPS:-5000}"
SIGMA_MODE="${SIGMA_MODE:-mid}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-8}"
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
    echo "Run: bash scripts/download_davis2017.sh" >&2
    exit 1
  fi

  local count
  count="$(find "${RAW_VIDEO_DIR}" -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.avi' \) | head -n 1 | wc -l)"
  if [[ "${count}" == "0" ]]; then
    echo "No raw videos found under: ${RAW_VIDEO_DIR}" >&2
    echo "Expected .mp4/.mov/.mkv/.webm/.avi files." >&2
    echo "Run: bash scripts/download_davis2017.sh" >&2
    echo "Or override RAW_VIDEO_DIR=/path/to/videos." >&2
    exit 1
  fi
}

build_latents() {
  check_paths
  check_raw_videos
  python "${PROJECT_ROOT}/scripts/build_latent_pairs.py" \
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
  if [[ ! -d "${LATENT_DIR}" ]] || [[ -z "$(find "${LATENT_DIR}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit 2>/dev/null)" ]]; then
    echo "No latent samples found under: ${LATENT_DIR}" >&2
    echo "Run build first: bash scripts/run_lightx2v_training.sh build" >&2
    exit 1
  fi

  local resume_args=()
  if [[ -n "${RESUME}" ]]; then
    resume_args=(--resume "${RESUME}")
  fi

  python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${CONFIG}" \
    --data_dir "${LATENT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --hidden_channels "${HIDDEN_CHANNELS}" \
    --num_res_blocks "${NUM_RES_BLOCKS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max_steps "${MAX_STEPS}" \
    --precision "${PRECISION}" \
    --sigma_mode "${SIGMA_MODE}" \
    --warmup_clean_steps "${WARMUP_CLEAN_STEPS}" \
    "${resume_args[@]}"
}

eval_latent() {
  python "${PROJECT_ROOT}/scripts/eval_latent.py" \
    --checkpoint "${OUT_DIR}/latest.pt" \
    --config "${CONFIG}" \
    --data_dir "${LATENT_DIR}" \
    --sigma_mode "${SIGMA_MODE}" \
    --precision "${PRECISION}" \
    --use_ema
}

case "${MODE}" in
  build)
    build_latents
    ;;
  train)
    train_model
    ;;
  eval)
    eval_latent
    ;;
  all)
    build_latents
    train_model
    ;;
  *)
    echo "Usage: bash scripts/run_lightx2v_training.sh [build|train|eval|all]" >&2
    exit 2
    ;;
esac
