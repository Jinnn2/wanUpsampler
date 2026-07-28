#!/usr/bin/env bash
set -euo pipefail

# End-to-end distilled SR chain:
#   LR Wan distill step1/2/base + step3 LoRA clean handoff
#   -> Stage2 clean latent resizer z_lr -> z_hr + re-noise
#   -> HR Wan distill step4/base
#   -> WAN VAE decode at HR.
#
# Useful overrides:
#   LORA_CKPT=/path/to/latest.safetensors
#   STAGE2_CHECKPOINT=/path/to/latest.pt
#   PROMPTS_FILE=changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt
#   LIMIT=10 CUDA_VISIBLE_DEVICES=0 bash changing_resolution_distill/scripts/eval/run_last_step_skip_lora_stage2_chain.sh
#
# For each prompt/seed this script writes five videos:
#   lora_stage2_after      : LR step3 LoRA handoff -> Stage2 -> HR step4
#   lora3_before_stage2    : LR step3 LoRA clean prediction decoded before Stage2
#   original3_stage2_after : LR step3 base handoff -> Stage2 -> HR step4
#   original3_before_stage2: LR step3 base clean prediction decoded before Stage2
#   original4_hr           : baseline HR 4-step distill without changing resolution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/path/to/LightX2V}"
MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-/path/to/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${MODEL_ROOT}/distill_model.pt}"
CR_DISTILL_LORA_OUT_DIR="${CR_DISTILL_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3}"
LORA_CKPT="${LORA_CKPT:-${CR_DISTILL_LORA_OUT_DIR}/latest.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"

CR_DISTILL_STAGE2_TAG="${CR_DISTILL_STAGE2_TAG:-14b_cfgdistill_5k}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_DISTILL_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_480p720p_stage2_${CR_DISTILL_STAGE2_TAG}_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_DISTILL_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml}}"

PROMPTS_FILE="${PROMPTS_FILE:-${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_lora_stage2_chain_720p}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-9600}"
LIMIT="${LIMIT:-8}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
INFER_STEPS="${INFER_STEPS:-4}"
CHANGE_STEP="${CHANGE_STEP:-3}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500 250}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5}"
RENOISE_MODE="${RENOISE_MODE:-random}"
STAGE2_USE_EMA="${STAGE2_USE_EMA:-0}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"

case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
done
for path in "${DIT_CKPT}" "${LORA_CKPT}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}" "${PROMPTS_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done
if (( CHANGE_STEP < 1 || CHANGE_STEP >= INFER_STEPS )); then
  echo "Invalid CHANGE_STEP=${CHANGE_STEP}; this chain requires at least one HR denoise step after the switch, so CHANGE_STEP must be in [1, $((INFER_STEPS - 1))]." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/configs" "${OUT_DIR}/videos" "${OUT_DIR}/compare"

mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

RATE="$(python -c "print(${LR_H} / ${HR_H})")"
BRIDGE_USE_EMA=false
if [[ "${STAGE2_USE_EMA}" == "1" ]]; then
  BRIDGE_USE_EMA=true
fi

write_config() {
  local output="$1"
  local case_name="$2"
  python - "$output" "$case_name" <<'PY'
import json
import os
import sys

path, case_name = sys.argv[1], sys.argv[2]
hr_h = int(os.environ["HR_H"])
hr_w = int(os.environ["HR_W"])
lr_h = int(os.environ["LR_H"])
lr_w = int(os.environ["LR_W"])
infer_steps = int(os.environ["INFER_STEPS"])
change_step = int(os.environ["CHANGE_STEP"])
denoising_steps = [int(x) for x in os.environ["DENOISING_STEP_LIST"].replace(",", " ").split()]

if case_name in {"lora3_before_stage2", "original3_before_stage2"}:
    target_h = lr_h
    target_w = lr_w
    infer_steps = change_step
    denoising_steps = denoising_steps[:change_step]
elif case_name in {"lora_stage2_after", "original3_stage2_after", "original4_hr"}:
    target_h = hr_h
    target_w = hr_w
else:
    raise SystemExit(f"unknown case_name: {case_name}")

cfg = {
    "infer_steps": infer_steps,
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": target_h,
    "target_width": target_w,
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
    "compare_name": case_name,
}

if case_name in {"lora_stage2_after", "original3_stage2_after"}:
    cfg.update(
        {
            "changing_resolution": True,
            "resolution_rate": [float(os.environ["RATE"])],
            "changing_resolution_steps": [change_step],
            "wan_distill_bridge_renoise_mode": os.environ["RENOISE_MODE"],
            "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
            "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
            "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
            "wan_clean_resizer_model_class": "stage2",
            "wan_clean_resizer_use_ema": os.environ["BRIDGE_USE_EMA"].lower() == "true",
        }
    )
    residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
    if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
        raise SystemExit("STAGE2_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")
    if residual_skip != "checkpoint":
        cfg["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

if case_name in {"lora_stage2_after", "lora3_before_stage2"}:
    cfg.update(
        {
            "lora_dynamic_apply": True,
            "lora_active_steps": [change_step],
            "lora_configs": [
                {
                    "name": "wan2.1",
                    "path": os.environ["LORA_CKPT"],
                    "strength": float(os.environ["LORA_STRENGTH"]),
                }
            ],
        }
    )
if case_name == "original3_before_stage2":
    cfg.update(
        {
            "lora_dynamic_apply": True,
            "lora_active_steps": [change_step],
            "lora_configs": [
                {
                    "name": "wan2.1",
                    "path": os.environ["LORA_CKPT"],
                    "strength": 0.0,
                }
            ],
        }
    )
if case_name in {"lora3_before_stage2", "original3_before_stage2"}:
    cfg["return_clean_pred_steps"] = [change_step]

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
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
    --save_result_path "${output}" \
    --target_video_length "${NUM_FRAMES}"
}

export PROJECT_ROOT DIT_CKPT LORA_CKPT LORA_STRENGTH STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG
export RATE BRIDGE_USE_EMA DENOISING_STEP_LIST INFER_STEPS NUM_FRAMES LR_H LR_W HR_H HR_W
export GUIDE_SCALE SAMPLE_SHIFT RENOISE_MODE CHANGE_STEP STAGE2_RESIDUAL_SKIP

echo "[lora-stage2] prompts=${PROMPTS_FILE} limit=${#prompts[@]} seed=${SEED}"
echo "[lora-stage2] lora=${LORA_CKPT} strength=${LORA_STRENGTH}"
echo "[lora-stage2] stage2=${STAGE2_CHECKPOINT}"
echo "[lora-stage2] size=${LR_H}x${LR_W} -> ${HR_H}x${HR_W} change_step=${CHANGE_STEP}->${INFER_STEPS}"

run_case() {
  local case_name="$1"
  local model_cls="$2"
  local sample_id="$3"
  local prompt="$4"
  local sample_seed="$5"
  local config_json="${OUT_DIR}/configs/${sample_id}_${case_name}.json"
  local output_dir="${OUT_DIR}/videos/${case_name}"
  local output="${output_dir}/${sample_id}_${case_name}.mp4"

  mkdir -p "${output_dir}"
  write_config "${config_json}" "${case_name}"
  echo "  [case] ${case_name} model_cls=${model_cls}"
  run_infer "${model_cls}" "${config_json}" "${prompt}" "${sample_seed}" "${output}"
}

make_compare() {
  local sample_id="$1"
  local out="${OUT_DIR}/compare/${sample_id}_lora_stage2_before_original3_before_original4_hstack.mp4"
  local v0="${OUT_DIR}/videos/lora_stage2_after/${sample_id}_lora_stage2_after.mp4"
  local v1="${OUT_DIR}/videos/lora3_before_stage2/${sample_id}_lora3_before_stage2.mp4"
  local v2="${OUT_DIR}/videos/original3_stage2_after/${sample_id}_original3_stage2_after.mp4"
  local v3="${OUT_DIR}/videos/original3_before_stage2/${sample_id}_original3_before_stage2.mp4"
  local v4="${OUT_DIR}/videos/original4_hr/${sample_id}_original4_hr.mp4"

  if ! command -v ffmpeg >/dev/null 2>&1; then
    return
  fi
  if [[ -f "${v0}" && -f "${v1}" && -f "${v2}" && -f "${v3}" && -f "${v4}" ]]; then
    ffmpeg -hide_banner -loglevel error -y \
      -i "${v0}" -i "${v1}" -i "${v2}" -i "${v3}" -i "${v4}" \
      -filter_complex "[0:v]scale=384:-2,setpts=PTS-STARTPTS[v0];[1:v]scale=384:-2,setpts=PTS-STARTPTS[v1];[2:v]scale=384:-2,setpts=PTS-STARTPTS[v2];[3:v]scale=384:-2,setpts=PTS-STARTPTS[v3];[4:v]scale=384:-2,setpts=PTS-STARTPTS[v4];[v0][v1][v2][v3][v4]hstack=inputs=5[v]" \
      -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${out}"
  else
    echo "[compare] skip hstack for ${sample_id}; missing one or more case videos" >&2
  fi
}

index=0
for prompt in "${prompts[@]}"; do
  global_index=$((PROMPT_OFFSET + index))
  sample_seed=$((SEED + global_index))
  sample_id="$(printf "%03d_seed%d" "${global_index}" "${sample_seed}")"

  echo "[$((index + 1))/${#prompts[@]}] ${sample_id}"
  echo "${prompt}"
  run_case "lora_stage2_after" "wan2.1_distill_last_step_lora_clean_resizer_bridge" "${sample_id}" "${prompt}" "${sample_seed}"
  run_case "lora3_before_stage2" "wan2.1_distill_last_step_lora" "${sample_id}" "${prompt}" "${sample_seed}"
  run_case "original3_stage2_after" "wan2.1_distill_clean_resizer_bridge" "${sample_id}" "${prompt}" "${sample_seed}"
  run_case "original3_before_stage2" "wan2.1_distill_last_step_lora" "${sample_id}" "${prompt}" "${sample_seed}"
  run_case "original4_hr" "wan2.1_distill" "${sample_id}" "${prompt}" "${sample_seed}"
  make_compare "${sample_id}"
  index=$((index + 1))
done

echo "LoRA/Stage2 comparison videos ready: ${OUT_DIR}/videos"
