#!/usr/bin/env bash
set -euo pipefail

# 验证目的：
# 1) 不加载训练好的 resizer，只用 interp bridge；
# 2) 对比 random vs resize_flow；
# 3) 如果 baseline 好、interp 都差，说明问题在 distill changing-resolution scheduler / handoff；
# 4) 如果 random 好、resize_flow 差，说明 resize_flow renoise 可疑。
#
# 用法：
#   bash changing_resolution_distill/scripts/bridge/01_run_interp_renoise_ab.sh
#
# 可覆盖变量：
#   STEP=2
#   MODES="random resize_flow"
#   PROMPT="..."

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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DTYPE="${DTYPE:-BF16}"
SEED="${SEED:-42}"
STEP="${STEP:-2}"
MODES="${MODES:-random resize_flow}"
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

for MODE in ${MODES}; do
  RUNTIME_CONFIG="${OUT_ROOT}/configs/interp_step${STEP}_${MODE}_seed${SEED}.json"
  OUT_VIDEO="${OUT_ROOT}/videos/01_interp_step${STEP}_${MODE}_seed${SEED}.mp4"

  python - "${CONFIG_TEMPLATE}" "${RUNTIME_CONFIG}" "${CR_DISTILL_DIT_CKPT}" "${HEIGHT}" "${WIDTH}" "${NUM_FRAMES}" "${STEP}" "${MODE}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, ckpt, height, width, num_frames, step, mode = sys.argv[1:]
data = json.loads(Path(src).read_text(encoding="utf-8"))

# interp bridge 不需要 clean resizer，删掉这些路径，避免 validate_config_paths 检查到无效 placeholder
for key in list(data.keys()):
    if key.startswith("wan_clean_resizer"):
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
    "changing_resolution": True,
    "resolution_rate": [2.0 / 3.0],
    "changing_resolution_steps": [int(step)],
    "wan_distill_bridge_renoise_mode": str(mode),
})

Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

  echo "[interp] step=${STEP} mode=${MODE}"
  echo "[interp] config=${RUNTIME_CONFIG}"
  echo "[interp] output=${OUT_VIDEO}"

  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
    --seed "${SEED}" \
    --model_cls "wan2.1_distill_interp_bridge" \
    --task "t2v" \
    --model_path "${CR_DISTILL_MODEL_ROOT}" \
    --config_json "${RUNTIME_CONFIG}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${OUT_VIDEO}" \
    --target_video_length "${NUM_FRAMES}"
done
