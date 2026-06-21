#!/usr/bin/env bash
set -euo pipefail

# LoRA strength sweep for one prompt/seed:
#   0.0, 0.1, 0.25, 0.5, 0.75, 1.0
#
# Usage:
#   bash changing_resolution_distill/scripts/eval/run_last_step_skip_lora_strength_sweep.sh

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
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
PROMPT="${PROMPT:-A cinematic night market street after rain, vendors cooking under warm lanterns, reflections on wet pavement, realistic crowd motion.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
STRENGTHS="${STRENGTHS:-0.0 0.1 0.25 0.5 0.75 1.0}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_strength_sweep}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_bridge_720p.example.json}"

if [[ ! -f "${CR_DISTILL_DIT_CKPT}" ]]; then
  echo "DiT checkpoint not found: ${CR_DISTILL_DIT_CKPT}" >&2
  exit 1
fi
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
  local strength="$1"
  local dst="$2"
  python - \
    "${CONFIG_TEMPLATE}" \
    "${dst}" \
    "${CR_DISTILL_DIT_CKPT}" \
    "${HEIGHT}" \
    "${WIDTH}" \
    "${NUM_FRAMES}" \
    "${LORA_CKPT}" \
    "${strength}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, ckpt, height, width, num_frames, lora_ckpt, strength = sys.argv[1:]
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
            "strength": float(strength),
        }
    ],
})
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

VIDEO_LIST=()
LABELS=()

for STRENGTH in ${STRENGTHS}; do
  TAG="${STRENGTH//./p}"
  CONFIG_PATH="${OUT_ROOT}/configs/lora3_strength_${TAG}_seed${SEED}.json"
  OUT_VIDEO="${OUT_ROOT}/videos/lora3_strength_${TAG}_seed${SEED}.mp4"
  make_config "${STRENGTH}" "${CONFIG_PATH}"

  echo "[sweep] strength=${STRENGTH}"
  echo "  config : ${CONFIG_PATH}"
  echo "  output : ${OUT_VIDEO}"

  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
    --seed "${SEED}" \
    --model_cls "wan2.1_distill_last_step_lora" \
    --task "t2v" \
    --model_path "${CR_DISTILL_MODEL_ROOT}" \
    --config_json "${CONFIG_PATH}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${OUT_VIDEO}" \
    --target_video_length "${NUM_FRAMES}"

  VIDEO_LIST+=("${OUT_VIDEO}")
  LABELS+=("s=${STRENGTH}")
done

if command -v ffmpeg >/dev/null 2>&1 && [[ "${#VIDEO_LIST[@]}" -eq 6 ]]; then
  STACKED="${OUT_ROOT}/videos/lora3_strength_sweep_seed${SEED}.mp4"
  echo "[sweep] creating 6-panel: ${STACKED}"
  ffmpeg -y \
    -i "${VIDEO_LIST[0]}" \
    -i "${VIDEO_LIST[1]}" \
    -i "${VIDEO_LIST[2]}" \
    -i "${VIDEO_LIST[3]}" \
    -i "${VIDEO_LIST[4]}" \
    -i "${VIDEO_LIST[5]}" \
    -filter_complex "[0:v]scale=426:-2,setpts=PTS-STARTPTS[v0];[1:v]scale=426:-2,setpts=PTS-STARTPTS[v1];[2:v]scale=426:-2,setpts=PTS-STARTPTS[v2];[3:v]scale=426:-2,setpts=PTS-STARTPTS[v3];[4:v]scale=426:-2,setpts=PTS-STARTPTS[v4];[5:v]scale=426:-2,setpts=PTS-STARTPTS[v5];[v0][v1][v2][v3][v4][v5]hstack=inputs=6[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${STACKED}"
else
  echo "[sweep] skip hstack: need ffmpeg and exactly 6 strengths"
fi
