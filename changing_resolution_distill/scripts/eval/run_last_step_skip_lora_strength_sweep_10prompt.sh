#!/usr/bin/env bash
set -euo pipefail

# 10-prompt LoRA strength sweep.
# The LoRA model is loaded once, then strengths/prompts are swept in-process.
#
# Strengths:
#   0.0, 0.1, 0.25, 0.5, 0.75, 1.0
#
# Usage:
#   bash changing_resolution_distill/scripts/eval/run_last_step_skip_lora_strength_sweep_10prompt.sh

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
CR_DISTILL_LORA_PLAN_D_OUT_DIR="${CR_DISTILL_LORA_PLAN_D_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_plan_d_rank16_qkvo_ffn}"
LORA_CKPT="${LORA_CKPT:-${CR_DISTILL_LORA_PLAN_D_OUT_DIR}/latest.safetensors}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DTYPE="${DTYPE:-BF16}"
SEED="${SEED:-42}"
INCREMENT_SEED="${INCREMENT_SEED:-1}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
LIMIT="${LIMIT:-10}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
STRENGTHS="${STRENGTHS:-0,0.1,0.25,0.5,0.75,1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_strength_sweep_10prompt}"
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

CONFIG_PATH="${OUT_ROOT}/configs/lora3_strength_sweep.json"

python - \
  "${CONFIG_TEMPLATE}" \
  "${CONFIG_PATH}" \
  "${CR_DISTILL_DIT_CKPT}" \
  "${HEIGHT}" \
  "${WIDTH}" \
  "${NUM_FRAMES}" \
  "${LORA_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, ckpt, height, width, num_frames, lora_ckpt = sys.argv[1:]
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

data.update({
    "infer_steps": 3,
    "target_video_length": int(num_frames),
    "target_height": int(height),
    "target_width": int(width),
    "sample_guide_scale": 6,
    "sample_shift": 5,
    "enable_cfg": False,
    "feature_caching": "NoCaching",
    "denoising_step_list": [1000, 750, 500],
    "dit_original_ckpt": str(ckpt),
    "lora_dynamic_apply": True,
    "lora_active_steps": [3],
    "lora_configs": [
        {
            "name": "wan2.1",
            "path": str(lora_ckpt),
            "strength": 0.0,
        }
    ],
})

Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

seed_args=()
if [[ "${INCREMENT_SEED}" == "1" ]]; then
  seed_args+=(--increment_seed)
fi

echo "[strength-10prompt] generating videos with one model load"
python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_strength_sweep_batch_infer.py" \
  --seed "${SEED}" \
  "${seed_args[@]}" \
  --model_cls "wan2.1_distill_last_step_lora" \
  --task "t2v" \
  --model_path "${CR_DISTILL_MODEL_ROOT}" \
  --config_json "${CONFIG_PATH}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --target_video_length "${NUM_FRAMES}" \
  --prompts_file "${PROMPTS_FILE}" \
  --out_dir "${OUT_ROOT}/videos" \
  --limit "${LIMIT}" \
  --strengths "${STRENGTHS}"

strength_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "${value}"
}

if command -v ffmpeg >/dev/null 2>&1; then
  IFS=',' read -r -a STRENGTH_ARRAY <<< "${STRENGTHS}"
  if [[ "${#STRENGTH_ARRAY[@]}" -eq 6 ]]; then
    for ((i = 0; i < LIMIT; i++)); do
      if [[ "${INCREMENT_SEED}" == "1" ]]; then
        item_seed=$((SEED + i))
      else
        item_seed="${SEED}"
      fi
      idx="$(printf "%02d" "${i}")"
      inputs=()
      filter_parts=()
      hstack_inputs=""
      for ((j = 0; j < 6; j++)); do
        tag="$(strength_tag "${STRENGTH_ARRAY[$j]}")"
        video="${OUT_ROOT}/videos/prompt${idx}_s${tag}_seed${item_seed}.mp4"
        inputs+=(-i "${video}")
        filter_parts+=("[${j}:v]scale=426:-2,setpts=PTS-STARTPTS[v${j}]")
        hstack_inputs+="[v${j}]"
      done
      stacked="${OUT_ROOT}/compare/prompt${idx}_seed${item_seed}_strength_sweep.mp4"
      if [[ -f "${OUT_ROOT}/videos/prompt${idx}_s$(strength_tag "${STRENGTH_ARRAY[0]}")_seed${item_seed}.mp4" && -f "${OUT_ROOT}/videos/prompt${idx}_s$(strength_tag "${STRENGTH_ARRAY[5]}")_seed${item_seed}.mp4" ]]; then
        filter_complex="$(IFS=';'; echo "${filter_parts[*]}");${hstack_inputs}hstack=inputs=6[v]"
        echo "[strength-10prompt] creating hstack prompt=${idx}: ${stacked}"
        ffmpeg -y \
          "${inputs[@]}" \
          -filter_complex "${filter_complex}" \
          -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${stacked}"
      else
        echo "[strength-10prompt] skip hstack prompt=${idx}; missing videos" >&2
      fi
    done
  else
    echo "[strength-10prompt] skip hstack: hstack script expects exactly 6 strengths, got ${#STRENGTH_ARRAY[@]}" >&2
  fi
else
  echo "[strength-10prompt] ffmpeg not found; skip hstack"
fi
