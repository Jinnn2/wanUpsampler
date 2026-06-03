#!/usr/bin/env bash
set -euo pipefail

# 验证目的：
# 1) 只测 step2 checkpoint；
# 2) 对比 random vs resize_flow；
# 3) 对比 EMA off vs EMA on；
# 4) 快速判断是 renoise 问题、EMA 问题，还是 resizer 本身问题。
#
# 用法：
#   bash changing_resolution_distill/scripts/bridge/02_run_step2_ckpt_renoise_ema_ab.sh
#
# 可覆盖变量：
#   STEP=2
#   CKPT=/path/to/latest.pt
#   EMA_LIST="0 1"
#   MODES="random resize_flow"

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
STEP="${STEP:-2}"
MODES="${MODES:-random resize_flow}"
EMA_LIST="${EMA_LIST:-0 1}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
PROMPT="${PROMPT:-A cinematic shot of a red sports car driving through a rainy city street at night, reflections on the road, smooth camera movement.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/distill_verify}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_bridge_720p.example.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_x0pred_480p_to_720p_lmdb_stage3_distill.yaml}"
CKPT="${CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step${STEP}_lmdb/latest.pt}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Set CKPT=/path/to/latest.pt or check CR_DISTILL_STAGE3_TAG=${CR_DISTILL_STAGE3_TAG}, STEP=${STEP}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos"

export CUDA_VISIBLE_DEVICES
export DTYPE
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

for MODE in ${MODES}; do
  for EMA in ${EMA_LIST}; do
    if [[ "${EMA}" == "1" ]]; then
      EMA_BOOL="true"
      EMA_TAG="ema"
    else
      EMA_BOOL="false"
      EMA_TAG="noema"
    fi

    RUNTIME_CONFIG="${OUT_ROOT}/configs/step${STEP}_${MODE}_${EMA_TAG}_seed${SEED}.json"
    OUT_VIDEO="${OUT_ROOT}/videos/02_step${STEP}_${MODE}_${EMA_TAG}_seed${SEED}.mp4"

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

    echo "[step2 ckpt] step=${STEP} mode=${MODE} ema=${EMA_TAG}"
    echo "[step2 ckpt] ckpt=${CKPT}"
    echo "[step2 ckpt] config=${RUNTIME_CONFIG}"
    echo "[step2 ckpt] output=${OUT_VIDEO}"

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
  done
done
