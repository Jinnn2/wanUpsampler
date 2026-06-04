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
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_stage3_steps_1_2_3_raw_ema_compare}"

MODEL_STEPS="${MODEL_STEPS:-1 2 3}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9500}"
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
for path in "${PROMPTS_FILE}" "${DIT_CKPT}" "${TRAIN_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

resolve_checkpoint_for_step() {
  local step="$1"
  local var_name="CHECKPOINT_STEP_${step}"
  local checkpoint="${!var_name-}"
  if [[ -n "${checkpoint}" ]]; then
    echo "${checkpoint}"
    return
  fi

  local checkpoint_dir="${PROJECT_ROOT}/outputs/changing_resolution_distill_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step${step}_lmdb"
  if [[ -f "${checkpoint_dir}/best_val.pt" ]]; then
    echo "${checkpoint_dir}/best_val.pt"
  else
    echo "${checkpoint_dir}/latest.pt"
  fi
}

MODEL_STEPS_NORMALIZED="${MODEL_STEPS//,/ }"
read -r -a model_steps <<< "${MODEL_STEPS_NORMALIZED}"
if [[ "${#model_steps[@]}" -ne 3 ]]; then
  echo "MODEL_STEPS must contain exactly three steps, for example: MODEL_STEPS='1 2 3' or MODEL_STEPS=1,2,3" >&2
  exit 2
fi
MODEL_STEP_TAG="${MODEL_STEPS_NORMALIZED// /_}"

for step in "${model_steps[@]}"; do
  if (( step < 1 || step > INFER_STEPS )); then
    echo "Invalid change step ${step}; must be in [1, ${INFER_STEPS}]." >&2
    exit 2
  fi

  var_name="CHECKPOINT_STEP_${step}"
  checkpoint="$(resolve_checkpoint_for_step "${step}")"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found for step ${step}: ${checkpoint}" >&2
    echo "Override with ${var_name}=/path/to/best_val.pt if needed." >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"/{configs,panels,compare}
for step in "${model_steps[@]}"; do
  mkdir -p "${OUT_DIR}/stage3_step${step}_raw" "${OUT_DIR}/stage3_step${step}_ema"
done

mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

RATE="$(python -c "print(${LR_H} / ${HR_H})")"

write_config() {
  local output="$1"
  local change_step="$2"
  local checkpoint="$3"
  local use_ema="$4"
  python - "$output" "$change_step" "$checkpoint" "$use_ema" <<'PY'
import json
import os
import sys

path = sys.argv[1]
change_step = int(sys.argv[2])
checkpoint = sys.argv[3]
use_ema = sys.argv[4].lower() == "true"
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
    "changing_resolution": True,
    "resolution_rate": [float(os.environ["RATE"])],
    "changing_resolution_steps": [change_step],
    "wan_distill_bridge_renoise_mode": os.environ["RENOISE_MODE"],
    "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
    "wan_clean_resizer_ckpt": checkpoint,
    "wan_clean_resizer_train_config": os.environ["TRAIN_CONFIG"],
    "wan_clean_resizer_model_class": "stage2",
    "wan_clean_resizer_use_ema": use_ema,
}

residual_skip = os.environ["STAGE3_RESIDUAL_SKIP"].lower()
if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
    raise SystemExit("STAGE3_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")
if residual_skip != "checkpoint":
    cfg["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
PY
}

run_infer() {
  local config_json="$1"
  local prompt="$2"
  local seed="$3"
  local output="$4"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${output}" ]]; then
    echo "skip existing: ${output}"
    return
  fi
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
    --seed "${seed}" \
    --model_cls "wan2.1_distill_clean_resizer_bridge" \
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

export PROJECT_ROOT TRAIN_CONFIG RATE DIT_CKPT DENOISING_STEP_LIST INFER_STEPS NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT HR_H HR_W RENOISE_MODE STAGE3_RESIDUAL_SKIP

index=0
for prompt in "${prompts[@]}"; do
  global_index=$((PROMPT_OFFSET + index))
  seed=$((START_SEED + global_index))
  sample_id="$(printf "%03d_seed%s" "${global_index}" "${seed}")"

  echo "[$((index + 1))/${#prompts[@]}] sample=${sample_id}"
  echo "${prompt}"

  panel_inputs=()
  for step in "${model_steps[@]}"; do
    checkpoint="$(resolve_checkpoint_for_step "${step}")"
    for variant in raw ema; do
      if [[ "${variant}" == "ema" ]]; then
        use_ema="true"
      else
        use_ema="false"
      fi

      cfg_stage3="${OUT_DIR}/configs/${sample_id}_stage3_step${step}_${variant}.json"
      out_stage3="${OUT_DIR}/stage3_step${step}_${variant}/${sample_id}_stage3_step${step}_${variant}.mp4"
      panel_stage3="${OUT_DIR}/panels/${sample_id}_panel_stage3_step${step}_${variant}.mp4"

      write_config "${cfg_stage3}" "${step}" "${checkpoint}" "${use_ema}"
      run_infer "${cfg_stage3}" "${prompt}" "${seed}" "${out_stage3}"
      make_labeled_panel "${out_stage3}" "${panel_stage3}" "stage3 step${step} ${variant}"
      panel_inputs+=("${panel_stage3}")
    done
  done

  compare="${OUT_DIR}/compare/${sample_id}_distill_steps_${MODEL_STEP_TAG}_raw_ema_compare.mp4"
  ffmpeg -hide_banner -loglevel error -y \
    -i "${panel_inputs[0]}" -i "${panel_inputs[1]}" -i "${panel_inputs[2]}" -i "${panel_inputs[3]}" -i "${panel_inputs[4]}" -i "${panel_inputs[5]}" \
    -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${compare}"

  index=$((index + 1))
done

echo "Distill step 1/2/3 raw-vs-EMA comparison videos ready: ${OUT_DIR}/compare"
