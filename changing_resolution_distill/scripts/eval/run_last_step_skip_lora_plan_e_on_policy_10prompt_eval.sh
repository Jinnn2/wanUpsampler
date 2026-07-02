#!/usr/bin/env bash
set -euo pipefail

# 10-prompt eval for Plan E on-policy teacher-trajectory LoRA.
#
# Model weights are loaded once per case:
#   1) original Wan distill, 3 denoise steps
#   2) original Wan distill, 3 denoise steps with LoRA enabled on configured steps
#   3) original Wan distill, 4 denoise steps
#
# Usage:
#   bash changing_resolution_distill/scripts/eval/run_last_step_skip_lora_plan_e_on_policy_10prompt_eval.sh
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=0
#   SEED=42
#   PROMPTS_FILE=changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt
#   LORA_CKPT=/path/to/latest.safetensors
#   LORA_ACTIVE_STEPS=3
#   LORA_CASE_NAME=01_lora3_step3_only
#   OUT_ROOT=outputs/changing_resolution_distill_teacher_trajectory_lora_plan_e_on_policy_10prompt_eval
#   HEIGHT=720 WIDTH=1248 NUM_FRAMES=81

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
CR_DISTILL_MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CR_DISTILL_DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${CR_DISTILL_MODEL_ROOT}/distill_model.pt}"
CR_DISTILL_TEACHER_TRAJ_OUT_DIR="${CR_DISTILL_TEACHER_TRAJ_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_teacher_trajectory_lora_plan_e_on_policy_velocity_rank16_qkvo_ffn}"
LORA_CKPT="${LORA_CKPT:-${CR_DISTILL_TEACHER_TRAJ_OUT_DIR}/latest.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"
LORA_ACTIVE_STEPS="${LORA_ACTIVE_STEPS:-3}"
LORA_CASE_NAME="${LORA_CASE_NAME:-01_lora3_step3_only}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DTYPE="${DTYPE:-BF16}"
SEED="${SEED:-42}"
INCREMENT_SEED="${INCREMENT_SEED:-1}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
LIMIT="${LIMIT:-10}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_teacher_trajectory_lora_plan_e_on_policy_10prompt_eval}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_bridge_720p.example.json}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"

if [[ ! -f "${CR_DISTILL_DIT_CKPT}" ]]; then
  echo "DiT checkpoint not found: ${CR_DISTILL_DIT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${LORA_CKPT}" ]]; then
  echo "LoRA checkpoint not found: ${LORA_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Prompts file not found: ${PROMPTS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos" "${OUT_ROOT}/compare"

export CUDA_VISIBLE_DEVICES
export DTYPE
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

make_config() {
  local name="$1"
  local infer_steps="$2"
  local denoise_steps="$3"
  local with_lora="$4"
  local lora_active_steps="$5"
  local dst="$6"

  python - \
    "${CONFIG_TEMPLATE}" \
    "${dst}" \
    "${CR_DISTILL_DIT_CKPT}" \
    "${HEIGHT}" \
    "${WIDTH}" \
    "${NUM_FRAMES}" \
    "${infer_steps}" \
    "${denoise_steps}" \
    "${with_lora}" \
    "${lora_active_steps}" \
    "${LORA_CKPT}" \
    "${LORA_STRENGTH}" \
    "${name}" <<'PY'
import json
import sys
from pathlib import Path

(
    src,
    dst,
    ckpt,
    height,
    width,
    num_frames,
    infer_steps,
    denoise_steps,
    with_lora,
    lora_active_steps,
    lora_ckpt,
    lora_strength,
    name,
) = sys.argv[1:]


def parse_step_list(text: str) -> list[int]:
    return [int(item) for item in text.replace(",", " ").split() if item.strip()]


data = json.loads(Path(src).read_text(encoding="utf-8"))
for key in list(data.keys()):
    if (
        key.startswith("wan_clean_resizer")
        or key in {
            "changing_resolution",
            "resolution_rate",
            "changing_resolution_steps",
            "wan_distill_bridge_renoise_mode",
        }
    ):
        data.pop(key, None)

steps = parse_step_list(denoise_steps)
data.update({
    "infer_steps": int(infer_steps),
    "target_video_length": int(num_frames),
    "target_height": int(height),
    "target_width": int(width),
    "sample_guide_scale": 6,
    "sample_shift": 5,
    "enable_cfg": False,
    "feature_caching": "NoCaching",
    "denoising_step_list": steps,
    "dit_original_ckpt": str(ckpt),
})

if with_lora == "1":
    active_steps = parse_step_list(lora_active_steps) or [int(infer_steps)]
    data.update({
        "lora_dynamic_apply": True,
        "lora_active_steps": active_steps,
        "lora_configs": [
            {
                "name": "wan2.1",
                "path": str(lora_ckpt),
                "strength": float(lora_strength),
            }
        ],
    })

data["compare_name"] = name
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_case_batch() {
  local name="$1"
  local model_cls="$2"
  local infer_steps="$3"
  local denoise_steps="$4"
  local with_lora="$5"
  local lora_active_steps="$6"

  local config_path="${OUT_ROOT}/configs/${name}.json"
  local case_out_dir="${OUT_ROOT}/videos/${name}"

  make_config "${name}" "${infer_steps}" "${denoise_steps}" "${with_lora}" "${lora_active_steps}" "${config_path}"
  mkdir -p "${case_out_dir}"

  echo "[plan-e-10prompt] case=${name}"
  echo "  model_cls         : ${model_cls}"
  echo "  config            : ${config_path}"
  echo "  out_dir           : ${case_out_dir}"
  if [[ "${with_lora}" == "1" ]]; then
    echo "  lora_ckpt         : ${LORA_CKPT}"
    echo "  lora_strength     : ${LORA_STRENGTH}"
    echo "  lora_active_steps : ${lora_active_steps}"
  fi

  local seed_args=()
  if [[ "${INCREMENT_SEED}" == "1" ]]; then
    seed_args+=(--increment_seed)
  fi

  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_batch_infer.py" \
    --seed "${SEED}" \
    "${seed_args[@]}" \
    --model_cls "${model_cls}" \
    --task "t2v" \
    --model_path "${CR_DISTILL_MODEL_ROOT}" \
    --config_json "${config_path}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --target_video_length "${NUM_FRAMES}" \
    --prompts_file "${PROMPTS_FILE}" \
    --out_dir "${case_out_dir}" \
    --name_prefix "${name}" \
    --limit "${LIMIT}"
}

run_case_batch "00_original3" "wan2.1_distill" "3" "1000,750,500" "0" ""
run_case_batch "${LORA_CASE_NAME}" "wan2.1_distill_last_step_lora" "3" "1000,750,500" "1" "${LORA_ACTIVE_STEPS}"
run_case_batch "02_original4" "wan2.1_distill" "4" "1000,750,500,250" "0" ""

if command -v ffmpeg >/dev/null 2>&1; then
  for ((i = 0; i < LIMIT; i++)); do
    if [[ "${INCREMENT_SEED}" == "1" ]]; then
      item_seed=$((SEED + i))
    else
      item_seed="${SEED}"
    fi
    idx="$(printf "%02d" "${i}")"
    v0="${OUT_ROOT}/videos/00_original3/00_original3_${idx}_seed${item_seed}.mp4"
    v1="${OUT_ROOT}/videos/${LORA_CASE_NAME}/${LORA_CASE_NAME}_${idx}_seed${item_seed}.mp4"
    v2="${OUT_ROOT}/videos/02_original4/02_original4_${idx}_seed${item_seed}.mp4"
    stacked="${OUT_ROOT}/compare/plan_e_eval_${idx}_seed${item_seed}_original3_lora3_original4.mp4"
    if [[ -f "${v0}" && -f "${v1}" && -f "${v2}" ]]; then
      echo "[plan-e-10prompt] creating hstack index=${idx}: ${stacked}"
      ffmpeg -y \
        -i "${v0}" \
        -i "${v1}" \
        -i "${v2}" \
        -filter_complex "[0:v]scale=640:-2,setpts=PTS-STARTPTS[v0];[1:v]scale=640:-2,setpts=PTS-STARTPTS[v1];[2:v]scale=640:-2,setpts=PTS-STARTPTS[v2];[v0][v1][v2]hstack=inputs=3[v]" \
        -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${stacked}"
    else
      echo "[plan-e-10prompt] skip hstack index=${idx}; missing one or more videos" >&2
    fi
  done
else
  echo "[plan-e-10prompt] ffmpeg not found; skip hstack"
fi
