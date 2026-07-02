#!/usr/bin/env bash
set -euo pipefail

# Generate ten controlled variants for each selected prompt:
# ORI480, Interp_step1/2/3, stage2_step1/2/3, stage3_step1/2/3.
#
# Outputs:
#   videos/       individual generated videos grouped by variant
#   panels/       labeled panels used for comparisons
#   compare/      full10, method sweep, and same-step control comparisons
#
# Common overrides:
#   LIMIT=10 PROMPT_OFFSET=0 START_SEED=9700
#   STAGE2_CHECKPOINT=/path/to/stage2.pt
#   CHECKPOINT_STAGE3_STEP_1=/path/to/stage3_step1.pt
#   CHECKPOINT_STAGE3_STEP_2=/path/to/stage3_step2.pt
#   CHECKPOINT_STAGE3_STEP_3=/path/to/stage3_step3.pt

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

CR_DISTILL_STAGE2_TAG="${CR_DISTILL_STAGE2_TAG:-14b_cfgdistill_5k}"
CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG:-14b_cfgdistill_5k}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_DISTILL_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml}}"
STAGE3_TRAIN_CONFIG="${STAGE3_TRAIN_CONFIG:-${CR_DISTILL_STAGE3_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_x0pred_480p_to_720p_lmdb_stage3_distill.yaml}}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_comprehensive_stage2_stage3_compare}"
STEPS="${STEPS:-1 2 3}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9700}"
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
STAGE2_USE_EMA="${STAGE2_USE_EMA:-0}"
STAGE3_USE_EMA="${STAGE3_USE_EMA:-${USE_EMA:-1}}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"
STAGE3_RESIDUAL_SKIP="${STAGE3_RESIDUAL_SKIP:-checkpoint}"
PANEL_SCALE_W="${PANEL_SCALE_W:-624}"
PANEL_SCALE_H="${PANEL_SCALE_H:-360}"
PANEL_LABEL_FONT_SIZE="${PANEL_LABEL_FONT_SIZE:-20}"
PANEL_LABEL_BOX_H="${PANEL_LABEL_BOX_H:-40}"

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

resolve_best_or_latest() {
  local checkpoint_dir="$1"
  if [[ -f "${checkpoint_dir}/best_val.pt" ]]; then
    echo "${checkpoint_dir}/best_val.pt"
  else
    echo "${checkpoint_dir}/latest.pt"
  fi
}

resolve_stage2_checkpoint() {
  if [[ -n "${STAGE2_CHECKPOINT:-}" ]]; then
    echo "${STAGE2_CHECKPOINT}"
    return
  fi
  if [[ -n "${CHECKPOINT_STAGE2:-}" ]]; then
    echo "${CHECKPOINT_STAGE2}"
    return
  fi
  local checkpoint_dir="${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_480p720p_stage2_${CR_DISTILL_STAGE2_TAG}_lmdb"
  resolve_best_or_latest "${checkpoint_dir}"
}

resolve_stage3_checkpoint_for_step() {
  local step="$1"
  local var_name="CHECKPOINT_STAGE3_STEP_${step}"
  local checkpoint="${!var_name-}"
  if [[ -n "${checkpoint}" ]]; then
    echo "${checkpoint}"
    return
  fi

  var_name="CHECKPOINT_STEP_${step}"
  checkpoint="${!var_name-}"
  if [[ -n "${checkpoint}" ]]; then
    echo "${checkpoint}"
    return
  fi

  local checkpoint_dir="${PROJECT_ROOT}/outputs/changing_resolution_distill_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step${step}_lmdb"
  resolve_best_or_latest "${checkpoint_dir}"
}

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
done
for path in "${PROMPTS_FILE}" "${DIT_CKPT}" "${STAGE2_TRAIN_CONFIG}" "${STAGE3_TRAIN_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

STEPS_NORMALIZED="${STEPS//,/ }"
read -r -a step_list <<< "${STEPS_NORMALIZED}"
if [[ "${#step_list[@]}" -ne 3 ]]; then
  echo "STEPS must contain exactly three steps, for example: STEPS='1 2 3' or STEPS=1,2,3" >&2
  exit 2
fi
for step in "${step_list[@]}"; do
  if (( step < 1 || step > INFER_STEPS )); then
    echo "Invalid step ${step}; must be in [1, ${INFER_STEPS}]." >&2
    exit 2
  fi
done

STAGE2_CHECKPOINT_RESOLVED="$(resolve_stage2_checkpoint)"
if [[ ! -f "${STAGE2_CHECKPOINT_RESOLVED}" ]]; then
  echo "Stage 2 checkpoint not found: ${STAGE2_CHECKPOINT_RESOLVED}" >&2
  echo "Override with STAGE2_CHECKPOINT=/path/to/best_val.pt if needed." >&2
  exit 1
fi

declare -A STAGE3_CHECKPOINTS
for step in "${step_list[@]}"; do
  checkpoint="$(resolve_stage3_checkpoint_for_step "${step}")"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Stage 3 checkpoint not found for step ${step}: ${checkpoint}" >&2
    echo "Override with CHECKPOINT_STAGE3_STEP_${step}=/path/to/best_val.pt if needed." >&2
    exit 1
  fi
  STAGE3_CHECKPOINTS["${step}"]="${checkpoint}"
done

mkdir -p "${OUT_DIR}"/{configs,videos,panels,compare}
mkdir -p "${OUT_DIR}/videos/ORI480"
for step in "${step_list[@]}"; do
  mkdir -p \
    "${OUT_DIR}/videos/Interp_step${step}" \
    "${OUT_DIR}/videos/stage2_step${step}" \
    "${OUT_DIR}/videos/stage3_step${step}"
done
mkdir -p \
  "${OUT_DIR}/compare/full10" \
  "${OUT_DIR}/compare/interp_sweep" \
  "${OUT_DIR}/compare/stage2_sweep" \
  "${OUT_DIR}/compare/stage3_sweep" \
  "${OUT_DIR}/compare/by_step"

mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

RATE="$(python -c "print(${LR_H} / ${HR_H})")"
STAGE2_BRIDGE_USE_EMA=false
if [[ "${STAGE2_USE_EMA}" == "1" ]]; then
  STAGE2_BRIDGE_USE_EMA=true
fi
STAGE3_BRIDGE_USE_EMA=false
if [[ "${STAGE3_USE_EMA}" == "1" ]]; then
  STAGE3_BRIDGE_USE_EMA=true
fi

write_config() {
  local output="$1"
  local mode="$2"
  local change_step="${3:-1}"
  local checkpoint="${4:-}"
  local train_config="${5:-}"
  local use_ema="${6:-false}"
  local residual_skip="${7:-checkpoint}"
  python - "$output" "$mode" "$change_step" "$checkpoint" "$train_config" "$use_ema" "$residual_skip" <<'PY'
import json
import os
import sys

path = sys.argv[1]
mode = sys.argv[2]
change_step = int(sys.argv[3])
checkpoint = sys.argv[4]
train_config = sys.argv[5]
use_ema = sys.argv[6].lower() == "true"
residual_skip = sys.argv[7].lower()
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

if mode == "ORI480":
    cfg.update({"target_height": int(os.environ["LR_H"]), "target_width": int(os.environ["LR_W"])})
elif mode in {"interp", "stage2", "stage3"}:
    cfg.update(
        {
            "changing_resolution": True,
            "resolution_rate": [float(os.environ["RATE"])],
            "changing_resolution_steps": [change_step],
            "wan_distill_bridge_renoise_mode": os.environ["RENOISE_MODE"],
        }
    )
    if mode in {"stage2", "stage3"}:
        if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
            raise SystemExit("residual_skip must be checkpoint, on/off, true/false, or 1/0")
        cfg.update(
            {
                "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
                "wan_clean_resizer_ckpt": checkpoint,
                "wan_clean_resizer_train_config": train_config,
                "wan_clean_resizer_model_class": "stage2",
                "wan_clean_resizer_use_ema": use_ema,
            }
        )
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
    --save_result_path "${output}" \
    --target_video_length "${NUM_FRAMES}"
}

make_labeled_panel() {
  local input="$1"
  local output="$2"
  local label="$3"
  ffmpeg -hide_banner -loglevel error -y -i "${input}" \
    -vf "scale=${PANEL_SCALE_W}:${PANEL_SCALE_H}:flags=bicubic,fps=${FPS},drawbox=x=0:y=0:w=iw:h=${PANEL_LABEL_BOX_H}:color=black@0.55:t=fill,drawtext=text='${label}':x=16:y=10:fontsize=${PANEL_LABEL_FONT_SIZE}:fontcolor=white" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${output}"
}

stack_h() {
  local output="$1"
  shift
  local inputs=("$@")
  local args=()
  local filter=""
  local index=0
  for input in "${inputs[@]}"; do
    args+=(-i "${input}")
    filter+="[${index}:v]"
    index=$((index + 1))
  done
  filter+="hstack=inputs=${#inputs[@]}[v]"
  ffmpeg -hide_banner -loglevel error -y \
    "${args[@]}" \
    -filter_complex "${filter}" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${output}"
}

stack_2x5() {
  local output="$1"
  shift
  local inputs=("$@")
  if [[ "${#inputs[@]}" -ne 10 ]]; then
    echo "stack_2x5 expects exactly 10 inputs, got ${#inputs[@]}" >&2
    exit 2
  fi

  ffmpeg -hide_banner -loglevel error -y \
    -i "${inputs[0]}" -i "${inputs[1]}" -i "${inputs[2]}" -i "${inputs[3]}" -i "${inputs[4]}" \
    -i "${inputs[5]}" -i "${inputs[6]}" -i "${inputs[7]}" -i "${inputs[8]}" -i "${inputs[9]}" \
    -filter_complex "[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[row0];[5:v][6:v][7:v][8:v][9:v]hstack=inputs=5[row1];[row0][row1]vstack=inputs=2[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${output}"
}

export PROJECT_ROOT RATE DIT_CKPT DENOISING_STEP_LIST INFER_STEPS NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT HR_H HR_W LR_H LR_W RENOISE_MODE

echo "Comprehensive distill control comparison"
echo "  prompts       : ${#prompts[@]} from ${PROMPTS_FILE}"
echo "  out_dir       : ${OUT_DIR}"
echo "  stage2_ckpt   : ${STAGE2_CHECKPOINT_RESOLVED}"
echo "  stage2_config : ${STAGE2_TRAIN_CONFIG}"
for step in "${step_list[@]}"; do
  echo "  stage3_step${step}: ${STAGE3_CHECKPOINTS[$step]}"
done

index=0
for prompt in "${prompts[@]}"; do
  global_index=$((PROMPT_OFFSET + index))
  seed=$((START_SEED + global_index))
  sample_id="$(printf "%03d_seed%d" "${global_index}" "${seed}")"

  echo "[$((index + 1))/${#prompts[@]}] sample=${sample_id}"
  echo "${prompt}"

  unset videos panels
  declare -A videos
  declare -A panels

  key="ORI480"
  cfg="${OUT_DIR}/configs/${sample_id}_${key}.json"
  video="${OUT_DIR}/videos/${key}/${sample_id}_${key}.mp4"
  panel="${OUT_DIR}/panels/${sample_id}_panel_${key}.mp4"
  write_config "${cfg}" "ORI480"
  run_infer "wan2.1_distill" "${cfg}" "${prompt}" "${seed}" "${video}"
  make_labeled_panel "${video}" "${panel}" "${key}"
  videos["${key}"]="${video}"
  panels["${key}"]="${panel}"

  for step in "${step_list[@]}"; do
    key="Interp_step${step}"
    cfg="${OUT_DIR}/configs/${sample_id}_${key}.json"
    video="${OUT_DIR}/videos/${key}/${sample_id}_${key}.mp4"
    panel="${OUT_DIR}/panels/${sample_id}_panel_${key}.mp4"
    write_config "${cfg}" "interp" "${step}"
    run_infer "wan2.1_distill_interp_bridge" "${cfg}" "${prompt}" "${seed}" "${video}"
    make_labeled_panel "${video}" "${panel}" "${key}"
    videos["${key}"]="${video}"
    panels["${key}"]="${panel}"
  done

  for step in "${step_list[@]}"; do
    key="stage2_step${step}"
    cfg="${OUT_DIR}/configs/${sample_id}_${key}.json"
    video="${OUT_DIR}/videos/${key}/${sample_id}_${key}.mp4"
    panel="${OUT_DIR}/panels/${sample_id}_panel_${key}.mp4"
    write_config \
      "${cfg}" "stage2" "${step}" \
      "${STAGE2_CHECKPOINT_RESOLVED}" "${STAGE2_TRAIN_CONFIG}" \
      "${STAGE2_BRIDGE_USE_EMA}" "${STAGE2_RESIDUAL_SKIP}"
    run_infer "wan2.1_distill_clean_resizer_bridge" "${cfg}" "${prompt}" "${seed}" "${video}"
    make_labeled_panel "${video}" "${panel}" "${key}"
    videos["${key}"]="${video}"
    panels["${key}"]="${panel}"
  done

  for step in "${step_list[@]}"; do
    key="stage3_step${step}"
    cfg="${OUT_DIR}/configs/${sample_id}_${key}.json"
    video="${OUT_DIR}/videos/${key}/${sample_id}_${key}.mp4"
    panel="${OUT_DIR}/panels/${sample_id}_panel_${key}.mp4"
    write_config \
      "${cfg}" "stage3" "${step}" \
      "${STAGE3_CHECKPOINTS[$step]}" "${STAGE3_TRAIN_CONFIG}" \
      "${STAGE3_BRIDGE_USE_EMA}" "${STAGE3_RESIDUAL_SKIP}"
    run_infer "wan2.1_distill_clean_resizer_bridge" "${cfg}" "${prompt}" "${seed}" "${video}"
    make_labeled_panel "${video}" "${panel}" "${key}"
    videos["${key}"]="${video}"
    panels["${key}"]="${panel}"
  done

  stack_2x5 \
    "${OUT_DIR}/compare/full10/${sample_id}_full10_ORI480_interp_stage2_stage3.mp4" \
    "${panels[ORI480]}" \
    "${panels[Interp_step${step_list[0]}]}" "${panels[Interp_step${step_list[1]}]}" "${panels[Interp_step${step_list[2]}]}" \
    "${panels[stage2_step${step_list[0]}]}" "${panels[stage2_step${step_list[1]}]}" "${panels[stage2_step${step_list[2]}]}" \
    "${panels[stage3_step${step_list[0]}]}" "${panels[stage3_step${step_list[1]}]}" "${panels[stage3_step${step_list[2]}]}"

  stack_h \
    "${OUT_DIR}/compare/interp_sweep/${sample_id}_ORI480_interp_steps.mp4" \
    "${panels[ORI480]}" \
    "${panels[Interp_step${step_list[0]}]}" "${panels[Interp_step${step_list[1]}]}" "${panels[Interp_step${step_list[2]}]}"

  stack_h \
    "${OUT_DIR}/compare/stage2_sweep/${sample_id}_ORI480_stage2_steps.mp4" \
    "${panels[ORI480]}" \
    "${panels[stage2_step${step_list[0]}]}" "${panels[stage2_step${step_list[1]}]}" "${panels[stage2_step${step_list[2]}]}"

  stack_h \
    "${OUT_DIR}/compare/stage3_sweep/${sample_id}_ORI480_stage3_steps.mp4" \
    "${panels[ORI480]}" \
    "${panels[stage3_step${step_list[0]}]}" "${panels[stage3_step${step_list[1]}]}" "${panels[stage3_step${step_list[2]}]}"

  for step in "${step_list[@]}"; do
    stack_h \
      "${OUT_DIR}/compare/by_step/${sample_id}_step${step}_interp_stage2_stage3.mp4" \
      "${panels[ORI480]}" \
      "${panels[Interp_step${step}]}" \
      "${panels[stage2_step${step}]}" \
      "${panels[stage3_step${step}]}"
  done

  index=$((index + 1))
done

echo "Comprehensive distill comparison videos ready:"
echo "  videos : ${OUT_DIR}/videos"
echo "  panels : ${OUT_DIR}/panels"
echo "  compare: ${OUT_DIR}/compare"
