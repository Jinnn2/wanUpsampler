#!/usr/bin/env bash
set -euo pipefail

# 验证目的：
# 1) 不走 changing-resolution，不走 upsampler；
# 2) 只看原始 Wan2.1 14B CfgDistill 720p 生成是否正常。
#
# 用法：
#   bash changing_resolution_distill/scripts/bridge/00_run_baseline_distill_720p.sh
#
# 可覆盖变量：
#   CUDA_VISIBLE_DEVICES=0
#   PROMPT="..."
#   SEED=42
#   OUT_ROOT=outputs/distill_verify
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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
DTYPE="${DTYPE:-BF16}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
PROMPT="${PROMPT:-A cinematic shot of a red sports car driving through a rainy city street at night, reflections on the road, smooth camera movement.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/distill_verify}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_bridge_720p.example.json}"

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos"

export CUDA_VISIBLE_DEVICES
export DTYPE
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

RUNTIME_CONFIG="${OUT_ROOT}/configs/baseline_distill_720p_seed${SEED}.json"
OUT_VIDEO="${OUT_ROOT}/videos/00_baseline_distill_720p_seed${SEED}.mp4"

python - "${CONFIG_TEMPLATE}" "${RUNTIME_CONFIG}" "${CR_DISTILL_DIT_CKPT}" "${HEIGHT}" "${WIDTH}" "${NUM_FRAMES}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, ckpt, height, width, num_frames = sys.argv[1:]
data = json.loads(Path(src).read_text(encoding="utf-8"))

# baseline 不走 bridge / changing resolution / resizer
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
    "infer_steps": 4,
    "target_video_length": int(num_frames),
    "target_height": int(height),
    "target_width": int(width),
    "sample_guide_scale": 6,
    "sample_shift": 5,
    "enable_cfg": False,
    "feature_caching": "NoCaching",
    "denoising_step_list": [1000, 750, 500, 250],
    "dit_original_ckpt": str(ckpt),
})

Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "[baseline] config=${RUNTIME_CONFIG}"
echo "[baseline] output=${OUT_VIDEO}"

python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
  --seed "${SEED}" \
  --model_cls "wan2.1_distill" \
  --task "t2v" \
  --model_path "${CR_DISTILL_MODEL_ROOT}" \
  --config_json "${RUNTIME_CONFIG}" \
  --prompt "${PROMPT}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --save_result_path "${OUT_VIDEO}" \
  --target_video_length "${NUM_FRAMES}"
