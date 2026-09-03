#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
if [[ "${MODE}" != "check" && "${MODE}" != "run" ]]; then
  echo "Usage: $0 [check|run]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
REALESRGAN_REPO="${REALESRGAN_REPO:-/mnt/afs_2/houze/Real-ESRGAN}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${PROJECT_ROOT}/UNIV_adaptor/configs/wan21_t2v_univ_rgb_720p.example.json}"
REALESRGAN_X2_CKPT="${REALESRGAN_X2_CKPT:-/mnt/afs_2/houze/Real-ESRGAN/weights/RealESRGAN_x2plus.pth}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/univ_adaptor_smoke}"
RESOLVED_CONFIG="${OUT_DIR}/resolved_config.json"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-${OUT_DIR}/sample.mp4}"

SPATIAL_RATIO="${SPATIAL_RATIO:-0.512}"
TEMPORAL_RATIO="${TEMPORAL_RATIO:-0.5}"
LR_NFE_RATIO="${LR_NFE_RATIO:-0.5}"
SWITCH_RATIO="${SWITCH_RATIO:-0.6}"
CACHE_MODE="${CACHE_MODE:-residual}"
TRANSITION_BASELINE="${TRANSITION_BASELINE:-rgb_sr_vae}"
NATIVE_HR_STATE_PATH="${NATIVE_HR_STATE_PATH:-}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-A cinematic shot of a red fox walking through a snowy forest.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

for path in "${WAN_PYTHON}" "${TEMPLATE_CONFIG}"; do
  [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
done
[[ -d "${MODEL_ROOT}" ]] || { echo "MODEL_ROOT not found: ${MODEL_ROOT}" >&2; exit 1; }
[[ -d "${LIGHTX2V_REPO}" ]] || { echo "Required directory not found: ${LIGHTX2V_REPO}" >&2; exit 1; }
if [[ "${TRANSITION_BASELINE}" == "rgb_sr_vae" ]]; then
  [[ -e "${REALESRGAN_X2_CKPT}" ]] || { echo "Required path not found: ${REALESRGAN_X2_CKPT}" >&2; exit 1; }
  [[ -d "${REALESRGAN_REPO}" ]] || { echo "Required directory not found: ${REALESRGAN_REPO}" >&2; exit 1; }
elif [[ "${TRANSITION_BASELINE}" != "dvg_latent_anchor" ]]; then
  echo "TRANSITION_BASELINE must be dvg_latent_anchor or rgb_sr_vae" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
"${WAN_PYTHON}" - \
  "${TEMPLATE_CONFIG}" "${RESOLVED_CONFIG}" "${REALESRGAN_X2_CKPT}" \
  "${SPATIAL_RATIO}" "${TEMPORAL_RATIO}" "${LR_NFE_RATIO}" \
  "${SWITCH_RATIO}" "${CACHE_MODE}" "${TRANSITION_BASELINE}" \
  "${NATIVE_HR_STATE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

template, output, checkpoint, rs, rt, rnfe, switch, cache_mode, transition, native_state = sys.argv[1:]
config = json.loads(Path(template).read_text(encoding="utf-8"))
config["univ_action"] = {
    "spatial_ratio": float(rs),
    "temporal_ratio": float(rt),
    "lr_nfe_ratio": float(rnfe),
    "switch_ratio": float(switch),
}
config["univ_cache_mode"] = cache_mode
config["univ_transition_baseline"] = transition
config["univ_native_hr_state_path"] = native_state
config["wan_rgb_sr_checkpoint"] = checkpoint
Path(output).write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

if [[ "${TRANSITION_BASELINE}" == "rgb_sr_vae" ]]; then
  export PYTHONPATH="${LIGHTX2V_REPO}:${REALESRGAN_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export LIGHTX2V_REPO
export DTYPE="${DTYPE:-BF16}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${WAN_PYTHON}" - "${RESOLVED_CONFIG}" "${MODEL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root
from UNIV_adaptor.schedule import action_from_config, resolve_schedule

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_config = validate_wan21_t2v_model_root(sys.argv[2])
action = action_from_config(config)
schedule = resolve_schedule(
    action,
    reference_nfe=int(config["infer_steps"]),
    target_latent_shape=(
        16,
        (int(config["target_video_length"]) - 1) // 4 + 1,
        int(config["target_height"]) // 8,
        int(config["target_width"]) // 8,
    ),
)
print(
    "Wan model contract passed: "
    f"dim={model_config['dim']}, num_heads={model_config['num_heads']}"
)
print(json.dumps(schedule.as_dict(), indent=2))
PY

"${WAN_PYTHON}" - "${TRANSITION_BASELINE}" <<'PY'
import sys
import torch

if sys.argv[1] == "rgb_sr_vae":
    from changing_resolution_distill.realesrgan_compat import install_functional_tensor_shim

    install_functional_tensor_shim()
    import basicsr  # noqa: F401
    import realesrgan  # noqa: F401
from lightx2v.common.ops import *  # noqa: F403
import UNIV_adaptor.wan_runner  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(f"Runtime imports passed; visible_cuda_devices={torch.cuda.device_count()}")
PY

if [[ "${MODE}" == "check" ]]; then
  echo "UNIV preflight passed; resolved config: ${RESOLVED_CONFIG}"
  exit 0
fi

"${WAN_PYTHON}" "${PROJECT_ROOT}/UNIV_adaptor/scripts/bridge/run_wan_univ_rgb_pipeline.py" \
  --model_cls wan2.1_univ_pipeline \
  --task t2v \
  --model_path "${MODEL_ROOT}" \
  --config_json "${RESOLVED_CONFIG}" \
  --prompt "${PROMPT}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --seed "${SEED}" \
  --target_video_length 81 \
  --save_result_path "${OUTPUT_VIDEO}"

echo "Video: ${OUTPUT_VIDEO}"
echo "Runtime metadata: ${OUTPUT_VIDEO}.univ.json"
