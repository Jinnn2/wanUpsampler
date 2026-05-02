#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/data/yongyang/Jin/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
RAW_VIDEO_DIR="${CR_RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p}"
CHECKPOINT="${CR_COMPARE_CKPT:-${CR_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p}/step_0100000.pt}"
TRAIN_CONFIG="${CR_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p.yaml}"
OUT_DIR="${CR_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_compare_step100000}"

LIMIT="${LIMIT:-10}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
FPS="${FPS:-16}"
PRECISION="${PRECISION:-bf16}"
USE_EMA="${USE_EMA:-1}"

export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LIGHTX2V_REPO

if [[ ! -d "${RAW_VIDEO_DIR}" ]]; then
  echo "Raw generated video dir not found: ${RAW_VIDEO_DIR}" >&2
  echo "Run: bash changing_resolution/scripts/run_clean_480p720p_training.sh generate" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Clean resizer checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "Train config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"/{ori480,ori720,interp720,trained720,compare}

mapfile -t videos < <(find "${RAW_VIDEO_DIR}" -type f -name '*.mp4' | sort | head -n "${LIMIT}")
if [[ "${#videos[@]}" -eq 0 ]]; then
  echo "No mp4 videos found under: ${RAW_VIDEO_DIR}" >&2
  exit 1
fi

make_labeled_panel() {
  local input="$1"
  local output="$2"
  local label="$3"
  local scale_flags="$4"
  ffmpeg -hide_banner -loglevel error -y -i "${input}" \
    -vf "scale=${HR_W}:${HR_H}:flags=${scale_flags},drawbox=x=0:y=0:w=iw:h=46:color=black@0.55:t=fill,drawtext=text='${label}':x=20:y=12:fontsize=24:fontcolor=white" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${output}"
}

index=0
for video in "${videos[@]}"; do
  sample_id="$(printf "%03d" "${index}")"
  stem="$(basename "${video}" .mp4)"
  ori480="${OUT_DIR}/ori480/${sample_id}_${stem}_ori480.mp4"
  ori720="${OUT_DIR}/ori720/${sample_id}_${stem}_ori720.mp4"
  interp720="${OUT_DIR}/interp720/${sample_id}_${stem}_interp720.mp4"
  trained720="${OUT_DIR}/trained720/${sample_id}_${stem}_trained720.mp4"
  p_ori480="${OUT_DIR}/compare/${sample_id}_${stem}_panel_ori480.mp4"
  p_ori720="${OUT_DIR}/compare/${sample_id}_${stem}_panel_ori720.mp4"
  p_interp="${OUT_DIR}/compare/${sample_id}_${stem}_panel_interp720.mp4"
  p_trained="${OUT_DIR}/compare/${sample_id}_${stem}_panel_trained720.mp4"
  compare="${OUT_DIR}/compare/${sample_id}_${stem}_compare.mp4"

  echo "[$((index + 1))/${#videos[@]}] ${video}"

  ffmpeg -hide_banner -loglevel error -y -i "${video}" \
    -vf "scale=${HR_W}:${HR_H}:flags=bicubic,fps=${FPS}" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${ori720}"

  ffmpeg -hide_banner -loglevel error -y -i "${ori720}" \
    -vf "scale=${LR_W}:${LR_H}:flags=bicubic" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${ori480}"

  ffmpeg -hide_banner -loglevel error -y -i "${ori480}" \
    -vf "scale=${HR_W}:${HR_H}:flags=bicubic" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${interp720}"

  ema_args=()
  if [[ "${USE_EMA}" == "1" ]]; then
    ema_args=(--use_ema)
  fi

  python "${PROJECT_ROOT}/changing_resolution/scripts/apply_clean_resizer_to_video.py" \
    --video_path "${ori720}" \
    --save_result_path "${trained720}" \
    --checkpoint "${CHECKPOINT}" \
    --train_config "${TRAIN_CONFIG}" \
    --model_root "${MODEL_ROOT}" \
    --vae_path "${VAE_PATH}" \
    --wan_repo "${LIGHTX2V_REPO}" \
    --vae_backend lightx2v \
    --precision "${PRECISION}" \
    --hr_size "${HR_H}" "${HR_W}" \
    --lr_size "${LR_H}" "${LR_W}" \
    --output_fps "${FPS}" \
    "${ema_args[@]}"

  make_labeled_panel "${ori480}" "${p_ori480}" "ori 480" "neighbor"
  make_labeled_panel "${ori720}" "${p_ori720}" "ori 720" "bicubic"
  make_labeled_panel "${interp720}" "${p_interp}" "interp 720" "bicubic"
  make_labeled_panel "${trained720}" "${p_trained}" "trained 720" "bicubic"

  ffmpeg -hide_banner -loglevel error -y \
    -i "${p_ori480}" -i "${p_ori720}" -i "${p_interp}" -i "${p_trained}" \
    -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${compare}"

  index=$((index + 1))
done

echo "Comparison videos ready: ${OUT_DIR}/compare"
