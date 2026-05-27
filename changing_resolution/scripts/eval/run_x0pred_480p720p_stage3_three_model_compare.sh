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
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_x0pred_480p_to_720p_lmdb_stage3.yaml}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_stage3_three_model_compare}"

MODEL_STEPS="${MODEL_STEPS:-45 46 47}"
INTERP_CHANGE_STEP="${INTERP_CHANGE_STEP:-45}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9200}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

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
USE_EMA="${USE_EMA:-0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
STAGE3_RESIDUAL_SKIP="${STAGE3_RESIDUAL_SKIP:-checkpoint}"

export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
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
for path in "${PROMPTS_FILE}" "${TRAIN_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

MODEL_STEPS_NORMALIZED="${MODEL_STEPS//,/ }"
read -r -a model_steps <<< "${MODEL_STEPS_NORMALIZED}"
if [[ "${#model_steps[@]}" -ne 3 ]]; then
  echo "MODEL_STEPS must contain exactly three steps, for example: MODEL_STEPS='45 46 47' or MODEL_STEPS=45,46,47" >&2
  exit 2
fi
MODEL_STEP_TAG="${MODEL_STEPS_NORMALIZED// /_}"

for step in "${INTERP_CHANGE_STEP}" "${model_steps[@]}"; do
  if (( step < 1 || step > INFER_STEPS )); then
    echo "Invalid change step ${step}; must be in [1, ${INFER_STEPS}]." >&2
    exit 2
  fi
done

for step in "${model_steps[@]}"; do
  var_name="CHECKPOINT_STEP_${step}"
  checkpoint="${!var_name-}"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="${PROJECT_ROOT}/outputs/changing_resolution_x0pred_480p720p_stage3_step${step}_lmdb/latest.pt"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found for step ${step}: ${checkpoint}" >&2
    echo "Override with ${var_name}=/path/to/latest.pt if needed." >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"/{configs,interp,panels,compare}
for step in "${model_steps[@]}"; do
  mkdir -p "${OUT_DIR}/stage3_step${step}"
done

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
  local change_step="$3"
  local checkpoint="${4:-}"
  python - "$output" "$mode" "$change_step" "$checkpoint" <<'PY'
import json
import os
import sys

path, mode, change_step, checkpoint = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

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
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "changing_resolution": True,
    "resolution_rate": [float(os.environ["RATE"])],
    "changing_resolution_steps": [change_step],
}

if mode == "stage3":
    residual_skip = os.environ["STAGE3_RESIDUAL_SKIP"].lower()
    if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
        raise SystemExit("STAGE3_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")
    cfg.update(
        {
            "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
            "wan_clean_resizer_ckpt": checkpoint,
            "wan_clean_resizer_train_config": os.environ["TRAIN_CONFIG"],
            "wan_clean_resizer_model_class": "stage2",
            "wan_clean_resizer_use_ema": os.environ["BRIDGE_USE_EMA"].lower() == "true",
        }
    )
    if residual_skip != "checkpoint":
        cfg["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}
elif mode != "interp":
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

export PROJECT_ROOT TRAIN_CONFIG RATE INFER_STEPS NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT BRIDGE_USE_EMA HR_H HR_W STAGE3_RESIDUAL_SKIP

index=0
for prompt in "${prompts[@]}"; do
  global_index=$((PROMPT_OFFSET + index))
  seed=$((START_SEED + global_index))
  sample_id="$(printf "%03d_seed%s" "${global_index}" "${seed}")"

  echo "[$((index + 1))/${#prompts[@]}] sample=${sample_id}"
  echo "${prompt}"

  cfg_interp="${OUT_DIR}/configs/${sample_id}_interp_step${INTERP_CHANGE_STEP}.json"
  out_interp="${OUT_DIR}/interp/${sample_id}_interp_step${INTERP_CHANGE_STEP}.mp4"
  panel_interp="${OUT_DIR}/panels/${sample_id}_panel_interp_step${INTERP_CHANGE_STEP}.mp4"
  write_config "${cfg_interp}" "interp" "${INTERP_CHANGE_STEP}"
  run_infer "wan2.1_clean_interp_bridge" "${cfg_interp}" "${prompt}" "${seed}" "${out_interp}"
  make_labeled_panel "${out_interp}" "${panel_interp}" "interp step ${INTERP_CHANGE_STEP}->${INFER_STEPS}"

  panel_inputs=("${panel_interp}")
  for step in "${model_steps[@]}"; do
    var_name="CHECKPOINT_STEP_${step}"
    checkpoint="${!var_name-}"
    if [[ -z "${checkpoint}" ]]; then
      checkpoint="${PROJECT_ROOT}/outputs/changing_resolution_x0pred_480p720p_stage3_step${step}_lmdb/latest.pt"
    fi
    cfg_stage3="${OUT_DIR}/configs/${sample_id}_stage3_step${step}.json"
    out_stage3="${OUT_DIR}/stage3_step${step}/${sample_id}_stage3_step${step}.mp4"
    panel_stage3="${OUT_DIR}/panels/${sample_id}_panel_stage3_step${step}.mp4"

    write_config "${cfg_stage3}" "stage3" "${step}" "${checkpoint}"
    run_infer "wan2.1_clean_resizer_bridge" "${cfg_stage3}" "${prompt}" "${seed}" "${out_stage3}"
    make_labeled_panel "${out_stage3}" "${panel_stage3}" "stage3 model step ${step}->${INFER_STEPS}"
    panel_inputs+=("${panel_stage3}")
  done

  compare="${OUT_DIR}/compare/${sample_id}_interp_vs_stage3_steps_${MODEL_STEP_TAG}.mp4"
  ffmpeg -hide_banner -loglevel error -y \
    -i "${panel_inputs[0]}" -i "${panel_inputs[1]}" -i "${panel_inputs[2]}" -i "${panel_inputs[3]}" \
    -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${compare}"

  index=$((index + 1))
done

echo "Stage 3 three-model comparison videos ready: ${OUT_DIR}/compare"
