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
MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${MODEL_ROOT}/distill_model.pt}"
PROMPTS_FILE="${PROMPTS_FILE:-${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}}"
CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG:-14b_cfgdistill_5k}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_x0pred_480p_to_720p_lmdb_stage3_distill.yaml}"
DEFAULT_CHECKPOINT_DIR="${PROJECT_ROOT}/outputs/changing_resolution_distill_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step2_lmdb"
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -f "${DEFAULT_CHECKPOINT_DIR}/best_val.pt" ]]; then
    CHECKPOINT="${DEFAULT_CHECKPOINT_DIR}/best_val.pt"
  else
    CHECKPOINT="${DEFAULT_CHECKPOINT_DIR}/latest.pt"
  fi
fi
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_stage3_step2_compare}"

LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9300}"
CHANGE_STEP="${CHANGE_STEP:-2}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
FPS="${FPS:-16}"
INFER_STEPS="${INFER_STEPS:-4}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500 250}"
PRECISION="${PRECISION:-bf16}"
USE_EMA="${USE_EMA:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
RENOISE_MODE="${RENOISE_MODE:-random}"
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
for path in "${PROMPTS_FILE}" "${DIT_CKPT}" "${CHECKPOINT}" "${TRAIN_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done
if (( CHANGE_STEP < 1 || CHANGE_STEP > INFER_STEPS )); then
  echo "Invalid CHANGE_STEP=${CHANGE_STEP}; must be in [1, ${INFER_STEPS}]." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"/{configs,low480,interp720,stage3_step2,panels,compare}

mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

RATE="$(python -c "print(${LR_H} / ${HR_H})")"
BRIDGE_USE_EMA=false
if [[ "${USE_EMA}" == "1" ]]; then
  BRIDGE_USE_EMA=true
fi

write_config() {
  local output="$1"
  local mode="$2"
  python - "$output" "$mode" <<'PY'
import json
import os
import sys

path, mode = sys.argv[1], sys.argv[2]
denoising_steps = [int(x) for x in os.environ["DENOISING_STEP_LIST"].replace(",", " ").split()]
cfg = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": int(os.environ["HR_H"]),
    "target_width": int(os.environ["HR_W"]),
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": float(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": False,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "denoising_step_list": denoising_steps,
    "dit_original_ckpt": os.environ["DIT_CKPT"],
}

if mode == "low480":
    cfg.update({"target_height": int(os.environ["LR_H"]), "target_width": int(os.environ["LR_W"])})
elif mode in {"interp", "stage3"}:
    cfg.update(
        {
            "changing_resolution": True,
            "resolution_rate": [float(os.environ["RATE"])],
            "changing_resolution_steps": [int(os.environ["CHANGE_STEP"])],
            "wan_distill_bridge_renoise_mode": os.environ["RENOISE_MODE"],
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
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
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

export PROJECT_ROOT CHECKPOINT TRAIN_CONFIG RATE DIT_CKPT DENOISING_STEP_LIST INFER_STEPS NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT BRIDGE_USE_EMA HR_H HR_W LR_H LR_W CHANGE_STEP RENOISE_MODE STAGE3_RESIDUAL_SKIP

index=0
for prompt in "${prompts[@]}"; do
  global_index=$((PROMPT_OFFSET + index))
  seed=$((START_SEED + global_index))
  sample_id="$(printf "%03d_step%02d" "${global_index}" "${CHANGE_STEP}")"

  echo "[$((index + 1))/${#prompts[@]}] sample=${sample_id} seed=${seed}"
  echo "${prompt}"

  cfg_low="${OUT_DIR}/configs/${sample_id}_low480.json"
  cfg_interp="${OUT_DIR}/configs/${sample_id}_interp720.json"
  cfg_stage3="${OUT_DIR}/configs/${sample_id}_stage3_step2.json"
  write_config "${cfg_low}" "low480"
  write_config "${cfg_interp}" "interp"
  write_config "${cfg_stage3}" "stage3"

  low480="${OUT_DIR}/low480/${sample_id}_low480.mp4"
  interp720="${OUT_DIR}/interp720/${sample_id}_interp720.mp4"
  stage3="${OUT_DIR}/stage3_step2/${sample_id}_stage3_step2.mp4"
  p_low="${OUT_DIR}/panels/${sample_id}_panel_low480.mp4"
  p_interp="${OUT_DIR}/panels/${sample_id}_panel_interp720.mp4"
  p_stage3="${OUT_DIR}/panels/${sample_id}_panel_stage3_step2.mp4"
  compare="${OUT_DIR}/compare/${sample_id}_distill_step2_compare.mp4"

  run_infer "wan2.1_distill" "${cfg_low}" "${prompt}" "${seed}" "${low480}"
  run_infer "wan2.1_distill_interp_bridge" "${cfg_interp}" "${prompt}" "${seed}" "${interp720}"
  run_infer "wan2.1_distill_clean_resizer_bridge" "${cfg_stage3}" "${prompt}" "${seed}" "${stage3}"

  make_labeled_panel "${low480}" "${p_low}" "no switch 480"
  make_labeled_panel "${interp720}" "${p_interp}" "interp 720 step ${CHANGE_STEP}->${INFER_STEPS}"
  make_labeled_panel "${stage3}" "${p_stage3}" "stage3 step2 model step ${CHANGE_STEP}->${INFER_STEPS}"
  ffmpeg -hide_banner -loglevel error -y \
    -i "${p_low}" -i "${p_interp}" -i "${p_stage3}" \
    -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${compare}"

  index=$((index + 1))
done

echo "Distill step-2 comparison videos ready: ${OUT_DIR}/compare"
