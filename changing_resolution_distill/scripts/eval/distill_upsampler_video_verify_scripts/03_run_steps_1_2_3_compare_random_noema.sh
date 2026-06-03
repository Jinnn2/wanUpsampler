#!/usr/bin/env bash
set -euo pipefail

# 验证目的：
# 1) 在已经确定较稳的 random + noEMA 条件下，横向对比 step1/2/3；
# 2) 避免 1_2_3 compare 混入 resize_flow 和 EMA 变量；
# 3) 如果 step2 明显最好，说明 handoff 位置是关键变量；
# 4) 如果三个都差，说明训练域/推理域错位或基础 bridge 仍有问题。
#
# 用法：
#   bash changing_resolution_distill/scripts/bridge/03_run_steps_1_2_3_compare_random_noema.sh
#
# 可覆盖变量：
#   STEPS="1 2 3"
#   MODE=random
#   EMA=0
#   CR_DISTILL_STAGE3_TAG=14b_cfgdistill_5k

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
CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG:-14b_cfgdistill_5k}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DTYPE="${DTYPE:-BF16}"
SEED="${SEED:-42}"
STEPS="${STEPS:-1 2 3}"
MODE="${MODE:-random}"
EMA="${EMA:-0}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
PROMPT="${PROMPT:-A cinematic shot of a red sports car driving through a rainy city street at night, reflections on the road, smooth camera movement.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/distill_verify}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_bridge_720p.example.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_x0pred_480p_to_720p_lmdb_stage3_distill.yaml}"

if [[ "${EMA}" == "1" ]]; then
  EMA_BOOL="true"
  EMA_TAG="ema"
else
  EMA_BOOL="false"
  EMA_TAG="noema"
fi

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos"

export CUDA_VISIBLE_DEVICES
export DTYPE
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

VIDEO_LIST=()

for STEP in ${STEPS}; do
  CKPT="${PROJECT_ROOT}/outputs/changing_resolution_distill_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step${STEP}_lmdb/latest.pt"
  if [[ ! -f "${CKPT}" ]]; then
    echo "Skip step=${STEP}; checkpoint not found: ${CKPT}" >&2
    continue
  fi

  RUNTIME_CONFIG="${OUT_ROOT}/configs/compare_step${STEP}_${MODE}_${EMA_TAG}_seed${SEED}.json"
  OUT_VIDEO="${OUT_ROOT}/videos/03_compare_step${STEP}_${MODE}_${EMA_TAG}_seed${SEED}.mp4"

  python - "${CONFIG_TEMPLATE}" "${RUNTIME_CONFIG}" "${CR_DISTILL_DIT_CKPT}" "${HEIGHT}" "${WIDTH}" "${NUM_FRAMES}" "${STEP}" "${MODE}" "${CKPT}" "${TRAIN_CONFIG}" "${PROJECT_ROOT}" "${EMA_BOOL}" <<'PY'
import json
import sys
from pathlib import Path

(
    src,
    dst,
    ckpt_dit,
    height,
    width,
    num_frames,
    step,
    mode,
    resizer_ckpt,
    train_config,
    repo,
    ema_bool,
) = sys.argv[1:]

data = json.loads(Path(src).read_text(encoding="utf-8"))
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
    "dit_original_ckpt": str(ckpt_dit),
    "changing_resolution": True,
    "resolution_rate": [2.0 / 3.0],
    "changing_resolution_steps": [int(step)],
    "wan_distill_bridge_renoise_mode": str(mode),
    "wan_clean_resizer_repo": str(repo),
    "wan_clean_resizer_ckpt": str(resizer_ckpt),
    "wan_clean_resizer_train_config": str(train_config),
    "wan_clean_resizer_model_class": "stage2",
    "wan_clean_resizer_residual_skip": False,
    "wan_clean_resizer_use_ema": ema_bool.lower() == "true",
})

Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

  echo "[compare] step=${STEP} mode=${MODE} ema=${EMA_TAG}"
  echo "[compare] ckpt=${CKPT}"
  echo "[compare] output=${OUT_VIDEO}"

  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
    --seed "${SEED}" \
    --model_cls "wan2.1_distill_clean_resizer_bridge" \
    --task "t2v" \
    --model_path "${CR_DISTILL_MODEL_ROOT}" \
    --config_json "${RUNTIME_CONFIG}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${OUT_VIDEO}" \
    --target_video_length "${NUM_FRAMES}"

  VIDEO_LIST+=("${OUT_VIDEO}")
done

if command -v ffmpeg >/dev/null 2>&1 && [[ "${#VIDEO_LIST[@]}" -eq 3 ]]; then
  STACKED="${OUT_ROOT}/videos/03_compare_steps_1_2_3_${MODE}_${EMA_TAG}_seed${SEED}_hstack.mp4"
  echo "[compare] creating hstack: ${STACKED}"
  ffmpeg -y \
    -i "${VIDEO_LIST[0]}" \
    -i "${VIDEO_LIST[1]}" \
    -i "${VIDEO_LIST[2]}" \
    -filter_complex "[0:v]scale=640:-2,setpts=PTS-STARTPTS[v0];[1:v]scale=640:-2,setpts=PTS-STARTPTS[v1];[2:v]scale=640:-2,setpts=PTS-STARTPTS[v2];[v0][v1][v2]hstack=inputs=3[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${STACKED}"
else
  echo "[compare] skip hstack: need ffmpeg and exactly 3 videos"
fi
