#!/usr/bin/env bash
set -euo pipefail

# Compare the mainline cached-x_pre_step3 LoRA objective across 10 prompts:
#   A. original3: base step1 + base step2 + base step3 clean prediction
#   B. lora3:     base step1 + base step2 + LoRA step3 clean prediction
#   C. teacher4:  base step1 + base step2 + base step3 + base step4
#
# Useful overrides:
#   PROMPTS_FILE=changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt
#   LIMIT=10
#   SEED=42
#   INCREMENT_SEED=1
#   LORA_CKPT=/path/to/step_0001000.safetensors
#
# 这个 eval 检查的是 Phase-1 LoRA 目标本身。LoRA LMDB 来自 480p/LR teacher
# trajectory，所以默认比较也必须跑在同一个 480p latent 分辨率上。只有在明确
# 想测试 out-of-distribution 720p 行为时，才手动覆盖 HEIGHT/WIDTH/CONFIG_TEMPLATE。

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
CR_DISTILL_LORA_OUT_DIR="${CR_DISTILL_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3}"
LORA_CKPT="${LORA_CKPT:-${CR_DISTILL_LORA_OUT_DIR}/latest.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DTYPE="${DTYPE:-BF16}"
SEED="${SEED:-42}"
INCREMENT_SEED="${INCREMENT_SEED:-1}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
LIMIT="${LIMIT:-10}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_clean_pred_compare_480p}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_480p.json}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"

for path in "${LIGHTX2V_REPO}" "${CR_DISTILL_MODEL_ROOT}" "${CR_DISTILL_DIT_CKPT}" "${CONFIG_TEMPLATE}" "${PROMPTS_FILE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Path not found: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${LORA_CKPT}" ]]; then
  echo "LoRA checkpoint not found: ${LORA_CKPT}" >&2
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
  local model_cls="$4"
  local lora_mode="$5"
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
    "${model_cls}" \
    "${lora_mode}" \
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
    model_cls,
    lora_mode,
    lora_ckpt,
    lora_strength,
    name,
) = sys.argv[1:]

data = json.loads(Path(src).read_text(encoding="utf-8"))
steps = [int(item) for item in denoise_steps.split(",") if item]
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
    "compare_name": name,
})

if model_cls == "wan2.1_distill_last_step_lora":
    strength = 0.0 if lora_mode == "off" else float(lora_strength)
    data.update({
        "lora_dynamic_apply": True,
        "lora_active_steps": [3],
        "return_clean_pred_steps": [3],
        "lora_configs": [
            {
                "name": "wan2.1",
                "path": str(lora_ckpt),
                "strength": strength,
            }
        ],
    })

Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_case_batch() {
  local name="$1"
  local model_cls="$2"
  local infer_steps="$3"
  local denoise_steps="$4"
  local lora_mode="$5"
  local config_path="${OUT_ROOT}/configs/${name}.json"
  local case_out_dir="${OUT_ROOT}/videos/${name}"

  make_config "${name}" "${infer_steps}" "${denoise_steps}" "${model_cls}" "${lora_mode}" "${config_path}"
  mkdir -p "${case_out_dir}"

  echo "[compare] ${name}"
  echo "  model_cls : ${model_cls}"
  echo "  config    : ${config_path}"
  echo "  out_dir   : ${case_out_dir}"

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

echo "[compare] prompts=${PROMPTS_FILE} limit=${LIMIT} seed=${SEED} increment_seed=${INCREMENT_SEED}"

run_case_batch "original3_clean_pred" "wan2.1_distill_last_step_lora" "3" "1000,750,500" "off"
run_case_batch "lora3_step3_clean_pred" "wan2.1_distill_last_step_lora" "3" "1000,750,500" "on"
run_case_batch "teacher4" "wan2.1_distill" "4" "1000,750,500,250" "none"

if command -v ffmpeg >/dev/null 2>&1; then
  for ((i = 0; i < LIMIT; i++)); do
    if [[ "${INCREMENT_SEED}" == "1" ]]; then
      item_seed=$((SEED + i))
    else
      item_seed="${SEED}"
    fi
    idx="$(printf "%02d" "${i}")"
    v0="${OUT_ROOT}/videos/original3_clean_pred/original3_clean_pred_${idx}_seed${item_seed}.mp4"
    v1="${OUT_ROOT}/videos/lora3_step3_clean_pred/lora3_step3_clean_pred_${idx}_seed${item_seed}.mp4"
    v2="${OUT_ROOT}/videos/teacher4/teacher4_${idx}_seed${item_seed}.mp4"
    stacked="${OUT_ROOT}/compare/${idx}_seed${item_seed}_original3_lora3_teacher4_hstack.mp4"
    if [[ -f "${v0}" && -f "${v1}" && -f "${v2}" ]]; then
      echo "[compare] creating hstack index=${idx}: ${stacked}"
      ffmpeg -y \
        -i "${v0}" \
        -i "${v1}" \
        -i "${v2}" \
        -filter_complex "[0:v]scale=640:-2,setpts=PTS-STARTPTS[v0];[1:v]scale=640:-2,setpts=PTS-STARTPTS[v1];[2:v]scale=640:-2,setpts=PTS-STARTPTS[v2];[v0][v1][v2]hstack=inputs=3[v]" \
        -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${stacked}"
    else
      echo "[compare] skip hstack index=${idx}; missing one or more videos" >&2
    fi
  done
else
  echo "[compare] ffmpeg not found; skip hstack"
fi
