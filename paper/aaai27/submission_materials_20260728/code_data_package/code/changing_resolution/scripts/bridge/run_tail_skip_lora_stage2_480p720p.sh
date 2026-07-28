#!/usr/bin/env bash
set -euo pipefail

# Single-path 50-step inference:
#   480x832 LR steps 1..44
#   -> step 45 with tail-skip LoRA
#   -> learned clean-latent resize to 720x1248
#   -> re-noise and HR steps 46..50
#   -> Wan VAE decode.

MODE="${1:-run}"
if [[ "${MODE}" != "run" && "${MODE}" != "check" ]]; then
  echo "Usage: $0 [run|check]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/path/to/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/Wan-AI/Wan2.1-T2V-1.3B}"

TRAIN_STEP="${TRAIN_STEP:-45}"
INFER_STEPS="${INFER_STEPS:-50}"
TAIL_SKIP_LORA_OUT_DIR="${TAIL_SKIP_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step${TRAIN_STEP}_to_step50}"
LORA_CKPT="${LORA_CKPT:-${TAIL_SKIP_LORA_OUT_DIR}/latest.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"

STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p_stage2_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml}}"
STAGE2_USE_EMA="${STAGE2_USE_EMA:-0}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"

LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-9700}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

PROMPTS_FILE="${PROMPTS_FILE:-${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}}"
PROMPT="${PROMPT:-}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step${TRAIN_STEP}_480p720p_single}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUT_DIR}/seed${SEED}_lora_stage2_720p.mp4}"
CONFIG_PATH="${CONFIG_PATH:-${OUT_DIR}/seed${SEED}_lora_stage2_480p720p.json}"

if [[ "${LR_H}" != "480" || "${LR_W}" != "832" || "${HR_H}" != "720" || "${HR_W}" != "1248" ]]; then
  echo "This entrypoint is locked to LR=480x832 and HR=720x1248." >&2
  echo "Received LR=${LR_H}x${LR_W}, HR=${HR_H}x${HR_W}." >&2
  exit 2
fi
if (( TRAIN_STEP < 1 || TRAIN_STEP >= INFER_STEPS )); then
  echo "Invalid TRAIN_STEP=${TRAIN_STEP}; expected [1, $((INFER_STEPS - 1))]." >&2
  exit 2
fi

case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16." >&2
    exit 2
    ;;
esac

if [[ -z "${PROMPT}" ]]; then
  if [[ ! -f "${PROMPTS_FILE}" ]]; then
    echo "PROMPT is empty and prompts file is missing: ${PROMPTS_FILE}" >&2
    exit 1
  fi
  PROMPT="$(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | head -n 1 || true)"
fi
if [[ -z "${PROMPT}" ]]; then
  echo "No prompt supplied. Set PROMPT or provide a non-empty PROMPTS_FILE." >&2
  exit 1
fi

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
done
for path in "${LORA_CKPT}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${OUTPUT_PATH}")" "$(dirname "${CONFIG_PATH}")"

export PROJECT_ROOT LORA_CKPT LORA_STRENGTH STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG
export TRAIN_STEP INFER_STEPS LR_H LR_W HR_H HR_W NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT
export STAGE2_USE_EMA STAGE2_RESIDUAL_SKIP

python - "${CONFIG_PATH}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
lr_h = int(os.environ["LR_H"])
hr_h = int(os.environ["HR_H"])
residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
    raise SystemExit("STAGE2_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")

config = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": int(os.environ["HR_H"]),
    "target_width": int(os.environ["HR_W"]),
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": float(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "changing_resolution": True,
    "resolution_rate": [lr_h / hr_h],
    "changing_resolution_steps": [int(os.environ["TRAIN_STEP"])],
    "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
    "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
    "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
    "wan_clean_resizer_model_class": "stage2",
    "wan_clean_resizer_use_ema": os.environ["STAGE2_USE_EMA"] == "1",
    "lora_dynamic_apply": True,
    "lora_active_steps": [int(os.environ["TRAIN_STEP"])],
    "lora_configs": [
        {
            "name": "wan2.1",
            "path": os.environ["LORA_CKPT"],
            "strength": float(os.environ["LORA_STRENGTH"]),
        }
    ],
}
if residual_skip != "checkpoint":
    config["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

echo "480p -> 720p full-chain configuration ready: ${CONFIG_PATH}"
echo "LoRA:  ${LORA_CKPT} (step=${TRAIN_STEP}, strength=${LORA_STRENGTH})"
echo "Stage2: ${STAGE2_CHECKPOINT} (ema=${STAGE2_USE_EMA})"
echo "Output: ${OUTPUT_PATH}"

if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; inference was not started."
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py" \
  --seed "${SEED}" \
  --model_cls wan2.1_tail_skip_lora_clean_resizer_bridge \
  --task t2v \
  --model_path "${MODEL_ROOT}" \
  --config_json "${CONFIG_PATH}" \
  --prompt "${PROMPT}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --save_result_path "${OUTPUT_PATH}" \
  --target_video_length "${NUM_FRAMES}"

echo "480p -> 720p full-chain video ready: ${OUTPUT_PATH}"
