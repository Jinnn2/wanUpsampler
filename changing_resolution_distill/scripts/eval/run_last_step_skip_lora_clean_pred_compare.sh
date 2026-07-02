#!/usr/bin/env bash
set -euo pipefail

# Compare the mainline cached-x_pre_step3 LoRA objective:
#   A. original3: base step1 + base step2 + base step3 clean prediction
#   B. lora3:     base step1 + base step2 + LoRA step3 clean prediction
#   C. teacher4:  base step1 + base step2 + base step3 + base step4

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
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
PROMPT="${PROMPT:-A cinematic shot of a red sports car driving through a rainy city street at night, reflections on the road, smooth camera movement.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_clean_pred_compare}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_generate_720p.json}"

for path in "${LIGHTX2V_REPO}" "${CR_DISTILL_MODEL_ROOT}" "${CR_DISTILL_DIT_CKPT}" "${CONFIG_TEMPLATE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Path not found: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${LORA_CKPT}" ]]; then
  echo "LoRA checkpoint not found: ${LORA_CKPT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos"

export CUDA_VISIBLE_DEVICES
export DTYPE
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

make_config() {
  local name="$1"
  local infer_steps="$2"
  local denoise_steps="$3"
  local model_cls="$4"
  local dst="$5"

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
    strength = 0.0 if name == "original3_clean_pred" else float(lora_strength)
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

run_case() {
  local name="$1"
  local model_cls="$2"
  local infer_steps="$3"
  local denoise_steps="$4"
  local config_path="${OUT_ROOT}/configs/${name}_seed${SEED}.json"
  local out_video="${OUT_ROOT}/videos/${name}_seed${SEED}.mp4"

  make_config "${name}" "${infer_steps}" "${denoise_steps}" "${model_cls}" "${config_path}"

  echo "[compare] ${name}"
  echo "  model_cls : ${model_cls}"
  echo "  config    : ${config_path}"
  echo "  output    : ${out_video}"

  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
    --seed "${SEED}" \
    --model_cls "${model_cls}" \
    --task "t2v" \
    --model_path "${CR_DISTILL_MODEL_ROOT}" \
    --config_json "${config_path}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${out_video}" \
    --target_video_length "${NUM_FRAMES}"
}

run_case "original3_clean_pred" "wan2.1_distill_last_step_lora" "3" "1000,750,500"
run_case "lora3_step3_clean_pred" "wan2.1_distill_last_step_lora" "3" "1000,750,500"
run_case "teacher4" "wan2.1_distill" "4" "1000,750,500,250"

if command -v ffmpeg >/dev/null 2>&1; then
  stacked="${OUT_ROOT}/videos/original3_lora3_teacher4_seed${SEED}_hstack.mp4"
  echo "[compare] creating hstack: ${stacked}"
  ffmpeg -y \
    -i "${OUT_ROOT}/videos/original3_clean_pred_seed${SEED}.mp4" \
    -i "${OUT_ROOT}/videos/lora3_step3_clean_pred_seed${SEED}.mp4" \
    -i "${OUT_ROOT}/videos/teacher4_seed${SEED}.mp4" \
    -filter_complex "[0:v]scale=640:-2,setpts=PTS-STARTPTS[v0];[1:v]scale=640:-2,setpts=PTS-STARTPTS[v1];[2:v]scale=640:-2,setpts=PTS-STARTPTS[v2];[v0][v1][v2]hstack=inputs=3[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${stacked}"
else
  echo "[compare] ffmpeg not found; skip hstack"
fi
