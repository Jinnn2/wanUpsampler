#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
PROMPTS_FILE="${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}"
CHECKPOINT="${CR_COMPARE_CKPT:-${CR_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p}/step_0100000.pt}"
TRAIN_CONFIG="${CR_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p.yaml}"
OUT_DIR="${CR_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_compare_step100000}"

LIMIT="${LIMIT:-10}"
START_SEED="${START_SEED:-9100}"
CHANGE_STEP="${CHANGE_STEP:-35}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
FPS="${FPS:-16}"
INFER_STEPS="${INFER_STEPS:-50}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
PRECISION="${PRECISION:-bf16}"
USE_EMA="${USE_EMA:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LIGHTX2V_REPO
case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16" >&2
    exit 2
    ;;
esac

if [[ ! -d "${LIGHTX2V_REPO}" ]]; then
  echo "LightX2V repo not found: ${LIGHTX2V_REPO}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "Wan model root not found: ${MODEL_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Prompts file not found: ${PROMPTS_FILE}" >&2
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

mkdir -p "${OUT_DIR}"/{configs,ori480,ori720,interp720,trained720,compare}

mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts found in: ${PROMPTS_FILE}" >&2
  exit 1
fi

RATE="$(python - <<PY
print(${LR_H} / ${HR_H})
PY
)"
BRIDGE_USE_EMA=false
if [[ "${USE_EMA}" == "1" ]]; then
  BRIDGE_USE_EMA=true
fi

write_config() {
  local output="$1"
  local height="$2"
  local width="$3"
  local mode="$4"
  python - "$output" "$height" "$width" "$mode" <<'PY'
import json
import os
import sys

path, height, width, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": height,
    "target_width": width,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": int(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
}
if mode in {"interp", "trained"}:
    cfg.update(
        {
            "changing_resolution": True,
            "resolution_rate": [float(os.environ["RATE"])],
            "changing_resolution_steps": [int(os.environ["CHANGE_STEP"])],
        }
    )
if mode == "trained":
    cfg.update(
        {
            "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
            "wan_clean_resizer_ckpt": os.environ["CHECKPOINT"],
            "wan_clean_resizer_train_config": os.environ["TRAIN_CONFIG"],
            "wan_clean_resizer_use_ema": os.environ["BRIDGE_USE_EMA"].lower() == "true",
        }
    )
with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
PY
}

export PROJECT_ROOT CHECKPOINT TRAIN_CONFIG RATE CHANGE_STEP INFER_STEPS NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT BRIDGE_USE_EMA

run_infer() {
  local model_cls="$1"
  local config_json="$2"
  local prompt="$3"
  local seed="$4"
  local output="$5"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${output}" ]]; then
    echo "skip existing: ${output}"
    return
  fi
  python "${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py" \
    --seed "${seed}" \
    --model_cls "${model_cls}" \
    --task t2v \
    --model_path "${MODEL_ROOT}" \
    --config_json "${config_json}" \
    --prompt "${prompt}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${output}"
}

make_labeled_panel() {
  local input="$1"
  local output="$2"
  local label="$3"
  ffmpeg -hide_banner -loglevel error -y -i "${input}" \
    -vf "scale=${HR_W}:${HR_H}:flags=bicubic,fps=${FPS},drawbox=x=0:y=0:w=iw:h=46:color=black@0.55:t=fill,drawtext=text='${label}':x=20:y=12:fontsize=24:fontcolor=white" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${output}"
}

index=0
for prompt in "${prompts[@]}"; do
  sample_id="$(printf "%03d" "${index}")"
  seed=$((START_SEED + index))

  cfg_480="${OUT_DIR}/configs/${sample_id}_ori480.json"
  cfg_720="${OUT_DIR}/configs/${sample_id}_ori720.json"
  cfg_interp="${OUT_DIR}/configs/${sample_id}_interp720.json"
  cfg_trained="${OUT_DIR}/configs/${sample_id}_trained720.json"
  write_config "${cfg_480}" "${LR_H}" "${LR_W}" "direct"
  write_config "${cfg_720}" "${HR_H}" "${HR_W}" "direct"
  write_config "${cfg_interp}" "${HR_H}" "${HR_W}" "interp"
  write_config "${cfg_trained}" "${HR_H}" "${HR_W}" "trained"

  ori480="${OUT_DIR}/ori480/${sample_id}_ori480.mp4"
  ori720="${OUT_DIR}/ori720/${sample_id}_ori720.mp4"
  interp720="${OUT_DIR}/interp720/${sample_id}_interp720.mp4"
  trained720="${OUT_DIR}/trained720/${sample_id}_trained720.mp4"
  p_ori480="${OUT_DIR}/compare/${sample_id}_panel_ori480.mp4"
  p_ori720="${OUT_DIR}/compare/${sample_id}_panel_ori720.mp4"
  p_interp="${OUT_DIR}/compare/${sample_id}_panel_interp720.mp4"
  p_trained="${OUT_DIR}/compare/${sample_id}_panel_trained720.mp4"
  compare="${OUT_DIR}/compare/${sample_id}_compare.mp4"

  echo "[$((index + 1))/${#prompts[@]}] seed=${seed}"
  echo "${prompt}"

  run_infer "wan2.1" "${cfg_480}" "${prompt}" "${seed}" "${ori480}"
  run_infer "wan2.1" "${cfg_720}" "${prompt}" "${seed}" "${ori720}"
  run_infer "wan2.1" "${cfg_interp}" "${prompt}" "${seed}" "${interp720}"
  run_infer "wan2.1_clean_resizer_bridge" "${cfg_trained}" "${prompt}" "${seed}" "${trained720}"

  make_labeled_panel "${ori480}" "${p_ori480}" "ori 480"
  make_labeled_panel "${ori720}" "${p_ori720}" "ori 720"
  make_labeled_panel "${interp720}" "${p_interp}" "interp 720"
  make_labeled_panel "${trained720}" "${p_trained}" "trained 720"

  ffmpeg -hide_banner -loglevel error -y \
    -i "${p_ori480}" -i "${p_ori720}" -i "${p_interp}" -i "${p_trained}" \
    -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${compare}"

  index=$((index + 1))
done

echo "Comparison videos ready: ${OUT_DIR}/compare"
