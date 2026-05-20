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
CHECKPOINT="${CR_STAGE3_CHANGE_STEP_SWEEP_CKPT:-${CR_STAGE3_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_x0pred_480p720p_stage3_lmdb}/latest.pt}"
TRAIN_CONFIG="${CR_STAGE3_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_x0pred_480p_to_720p_lmdb_stage3.yaml}"
OUT_DIR="${CR_STAGE3_CHANGE_STEP_SWEEP_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_stage3_change_step_sweep}"

LIMIT="${LIMIT:-1}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9100}"
STEP_START="${STEP_START:-10}"
STEP_END="${STEP_END:-50}"
STEP_STRIDE="${STEP_STRIDE:-5}"
CHANGE_STEPS="${CHANGE_STEPS:-35}"
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
# Stage 3 is trained for the x0_pred handoff domain; the 50k run should have a usable EMA.
USE_EMA="${USE_EMA:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
STAGE3_RESIDUAL_SKIP="${STAGE3_RESIDUAL_SKIP:-checkpoint}"

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

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
done
for path in "${PROMPTS_FILE}" "${CHECKPOINT}" "${TRAIN_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"/{configs,stop480,interp720,stage3_720,compare}

mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

if [[ -n "${CHANGE_STEPS}" ]]; then
  read -r -a steps <<< "${CHANGE_STEPS}"
else
  steps=()
  step="${STEP_START}"
  while (( step <= STEP_END )); do
    steps+=("${step}")
    step=$((step + STEP_STRIDE))
  done
fi
if [[ "${#steps[@]}" -eq 0 ]]; then
  echo "No change steps selected." >&2
  exit 1
fi
for step in "${steps[@]}"; do
  if (( step < 1 || step > INFER_STEPS )); then
    echo "Invalid change step ${step}; must be in [1, ${INFER_STEPS}]." >&2
    exit 1
  fi
done

RATE="$(python -c "print(${LR_H} / ${HR_H})")"
BRIDGE_USE_EMA=false
if [[ "${USE_EMA}" == "1" ]]; then
  BRIDGE_USE_EMA=true
fi

write_config() {
  local output="$1"
  local mode="$2"
  local change_step="$3"
  python - "$output" "$mode" "$change_step" <<'PY'
import json
import os
import sys

path, mode, change_step = sys.argv[1], sys.argv[2], int(sys.argv[3])
hr_h = int(os.environ["HR_H"])
hr_w = int(os.environ["HR_W"])
lr_h = int(os.environ["LR_H"])
lr_w = int(os.environ["LR_W"])

cfg = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": hr_h,
    "target_width": hr_w,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": int(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
}

if mode == "stop480":
    cfg.update(
        {
            "target_height": lr_h,
            "target_width": lr_w,
            "stop_after_steps": change_step,
        }
    )
elif mode in {"interp", "stage3"}:
    cfg.update(
        {
            "changing_resolution": True,
            "resolution_rate": [float(os.environ["RATE"])],
            "changing_resolution_steps": [change_step],
        }
    )
    if mode == "stage3":
        cfg.update(
            {
                "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
                "wan_clean_resizer_ckpt": os.environ["CHECKPOINT"],
                "wan_clean_resizer_train_config": os.environ["TRAIN_CONFIG"],
                "wan_clean_resizer_model_class": "stage2",
                "wan_clean_resizer_use_ema": os.environ["BRIDGE_USE_EMA"].lower() == "true",
            }
        )
        residual_skip = os.environ["STAGE3_RESIDUAL_SKIP"].lower()
        if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
            raise SystemExit("STAGE3_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")
        if residual_skip != "checkpoint":
            cfg["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}
else:
    raise SystemExit(f"unknown mode: {mode}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
PY
}

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

export PROJECT_ROOT CHECKPOINT TRAIN_CONFIG RATE INFER_STEPS NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT BRIDGE_USE_EMA HR_H HR_W LR_H LR_W STAGE3_RESIDUAL_SKIP

index=0
for prompt in "${prompts[@]}"; do
  global_index=$((PROMPT_OFFSET + index))
  seed=$((START_SEED + global_index))

  for change_step in "${steps[@]}"; do
    sample_id="$(printf "%03d_step%02d" "${global_index}" "${change_step}")"
    cfg_stop="${OUT_DIR}/configs/${sample_id}_stop480.json"
    cfg_interp="${OUT_DIR}/configs/${sample_id}_interp720.json"
    cfg_stage3="${OUT_DIR}/configs/${sample_id}_stage3_720.json"
    write_config "${cfg_stop}" "stop480" "${change_step}"
    write_config "${cfg_interp}" "interp" "${change_step}"
    write_config "${cfg_stage3}" "stage3" "${change_step}"

    stop480="${OUT_DIR}/stop480/${sample_id}_stop480.mp4"
    interp720="${OUT_DIR}/interp720/${sample_id}_interp720.mp4"
    stage3_720="${OUT_DIR}/stage3_720/${sample_id}_stage3_720.mp4"
    p_stop="${OUT_DIR}/compare/${sample_id}_panel_stop480.mp4"
    p_interp="${OUT_DIR}/compare/${sample_id}_panel_interp720.mp4"
    p_stage3="${OUT_DIR}/compare/${sample_id}_panel_stage3_720.mp4"
    compare="${OUT_DIR}/compare/${sample_id}_stage3_step_sweep_compare.mp4"

    echo "[$((index + 1))/${#prompts[@]}] global=${global_index} seed=${seed} step=${change_step}/${INFER_STEPS}"
    echo "${prompt}"

    run_infer "wan2.1_partial_denoise_decode" "${cfg_stop}" "${prompt}" "${seed}" "${stop480}"
    run_infer "wan2.1_clean_interp_bridge" "${cfg_interp}" "${prompt}" "${seed}" "${interp720}"
    run_infer "wan2.1_clean_resizer_bridge" "${cfg_stage3}" "${prompt}" "${seed}" "${stage3_720}"

    make_labeled_panel "${stop480}" "${p_stop}" "stop 480 step ${change_step}"
    make_labeled_panel "${interp720}" "${p_interp}" "interp 720 step ${change_step}->${INFER_STEPS}"
    make_labeled_panel "${stage3_720}" "${p_stage3}" "stage3 720 step ${change_step}->${INFER_STEPS}"
    ffmpeg -hide_banner -loglevel error -y \
      -i "${p_stop}" -i "${p_interp}" -i "${p_stage3}" \
      -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
      -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${compare}"
  done

  index=$((index + 1))
done

echo "Stage 3 change-step sweep videos ready: ${OUT_DIR}/compare"
