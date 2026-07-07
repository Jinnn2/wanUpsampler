#!/usr/bin/env bash
set -euo pipefail

# Generates two direct-decode probes:
#   1) Wan distill LoRA -> Stage2 clean latent resize -> VAE decode, with no HR denoise step.
#   2) LTX2 8-step t2av -> LTX2 latent upsampler -> VAE decode, with no Stage2 denoise loop.
#
# Typical use:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/test/run_stage2_direct_decode_compare.sh
#
# Useful overrides:
#   RUN_DISTILL=1 RUN_LTX2=1
#   PROMPT="A beautiful sunset over the ocean"
#   OUT_DIR=outputs/test_stage2_direct_decode
#   LORA_CKPT=/path/to/latest.safetensors
#   STAGE2_CHECKPOINT=/path/to/latest.pt
#   LIGHTX2V_REPO=/path/to/LightX2V
#   LTX2_MODEL_PATH=Lightricks/LTX-2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/test_stage2_direct_decode}"
DEFAULT_LIGHTX2V_REPO=""
if [[ -d "${PROJECT_ROOT}/../LightX2V" ]]; then
  DEFAULT_LIGHTX2V_REPO="$(cd "${PROJECT_ROOT}/../LightX2V" && pwd)"
  LIGHTX2V_REPO="${LIGHTX2V_REPO:-${DEFAULT_LIGHTX2V_REPO}}"
fi

cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-${DEFAULT_LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}}"
MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-${MODEL_ROOT:-/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}}"
DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${MODEL_ROOT}/distill_model.pt}"

CR_DISTILL_LORA_OUT_DIR="${CR_DISTILL_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3}"
LORA_CKPT="${LORA_CKPT:-${CR_DISTILL_LORA_OUT_DIR}/latest.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"

CR_DISTILL_STAGE2_TAG="${CR_DISTILL_STAGE2_TAG:-14b_cfgdistill_5k}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_DISTILL_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_480p720p_stage2_${CR_DISTILL_STAGE2_TAG}_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_DISTILL_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml}}"

PROMPTS_FILE="${PROMPTS_FILE:-${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}}"
PROMPT="${PROMPT:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-9600}"
RUN_DISTILL="${RUN_DISTILL:-1}"
RUN_LTX2="${RUN_LTX2:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
CHANGE_STEP="${CHANGE_STEP:-3}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5}"
STAGE2_USE_EMA="${STAGE2_USE_EMA:-0}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"

LTX2_MODEL_PATH="${LTX2_MODEL_PATH:-Lightricks/LTX-2}"
LTX2_CONFIG_JSON="${LTX2_CONFIG_JSON:-${LIGHTX2V_REPO}/configs/ltx2/ltx2_upsample.json}"
LTX2_NUM_FRAMES="${LTX2_NUM_FRAMES:-121}"
LTX2_NEGATIVE_PROMPT="${LTX2_NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"

case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16" >&2
    exit 2
    ;;
esac
REQUESTED_DTYPE="${DTYPE}"

export CUDA_VISIBLE_DEVICES
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}/configs" "${OUT_DIR}/videos" "${OUT_DIR}/tmp"

if [[ -z "${PROMPT}" ]]; then
  if [[ ! -f "${PROMPTS_FILE}" ]]; then
    echo "PROMPT is empty and prompts file not found: ${PROMPTS_FILE}" >&2
    exit 1
  fi
  PROMPT="$(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | head -n 1)"
fi
if [[ -z "${PROMPT}" ]]; then
  echo "No prompt selected. Set PROMPT=... or provide PROMPTS_FILE." >&2
  exit 1
fi

check_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
}

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
}

write_distill_config() {
  local output="$1"
  python - "$output" <<'PY'
import json
import os
import sys

path = sys.argv[1]
change_step = int(os.environ["CHANGE_STEP"])
denoising_steps = [int(x) for x in os.environ["DENOISING_STEP_LIST"].replace(",", " ").split()]
if len(denoising_steps) < change_step:
    raise SystemExit("DENOISING_STEP_LIST must contain at least CHANGE_STEP entries")

residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
    raise SystemExit("STAGE2_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")

cfg = {
    "infer_steps": change_step,
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": int(os.environ["HR_H"]),
    "target_width": int(os.environ["HR_W"]),
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": float(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": False,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "denoising_step_list": denoising_steps[:change_step],
    "dit_original_ckpt": os.environ["DIT_CKPT"],
    "compare_name": "distill_lora_stage2_direct_decode",
    "changing_resolution": True,
    "resolution_rate": [float(os.environ["LR_H"]) / float(os.environ["HR_H"])],
    "changing_resolution_steps": [change_step],
    "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
    "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
    "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
    "wan_clean_resizer_model_class": "stage2",
    "wan_clean_resizer_use_ema": os.environ["STAGE2_USE_EMA"] == "1",
    "lora_dynamic_apply": True,
    "lora_active_steps": [change_step],
    "lora_configs": [
        {
            "name": "wan2.1",
            "path": os.environ["LORA_CKPT"],
            "strength": float(os.environ["LORA_STRENGTH"]),
        }
    ],
}
if residual_skip != "checkpoint":
    cfg["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

run_distill_direct_decode() {
  check_dir "${LIGHTX2V_REPO}"
  check_dir "${MODEL_ROOT}"
  check_file "${DIT_CKPT}"
  check_file "${LORA_CKPT}"
  check_file "${STAGE2_CHECKPOINT}"
  check_file "${STAGE2_TRAIN_CONFIG}"

  local config_json="${OUT_DIR}/configs/distill_lora_stage2_direct_decode.json"
  local output="${OUT_DIR}/videos/distill_lora_stage2_direct_decode_seed${SEED}.mp4"

  export PROJECT_ROOT DIT_CKPT LORA_CKPT LORA_STRENGTH STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG
  export DENOISING_STEP_LIST NUM_FRAMES LR_H LR_W HR_H HR_W GUIDE_SCALE SAMPLE_SHIFT CHANGE_STEP
  export STAGE2_USE_EMA STAGE2_RESIDUAL_SKIP

  write_distill_config "${config_json}"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${output}" ]]; then
    echo "[distill] skip existing: ${output}"
    return
  fi

  echo "[distill] LoRA -> Stage2 -> direct VAE decode"
  echo "[distill] output=${output}"
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py" \
    --seed "${SEED}" \
    --model_cls "wan2.1_distill_last_step_lora_clean_resizer_bridge" \
    --task t2v \
    --model_path "${MODEL_ROOT}" \
    --config_json "${config_json}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${output}" \
    --target_video_length "${NUM_FRAMES}"
}

write_ltx2_direct_wrapper() {
  local output="$1"
  cat > "${output}" <<'PY'
import gc

import torch
from loguru import logger

from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner
from lightx2v_platform.base.global_var import AI_DEVICE


def run_upsampler_direct_decode(self, v_latent, a_latent):
    logger.info("LTX2 Stage 2 direct decode probe: upsample latent and skip Stage2 denoise loop.")
    if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
        self.upsampler = self.load_upsampler()

    upsampled_v_latent = self.upsampler.upsample(v_latent, self.video_vae.encoder).squeeze(0)

    if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
        del self.upsampler
        getattr(torch, AI_DEVICE).empty_cache()
        gc.collect()

    self.input_info.target_shape = [
        self.input_info.target_shape[0] * 2,
        self.input_info.target_shape[1] * 2,
    ]
    self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
    self._clear_ltx2_reference_video_state()
    return upsampled_v_latent, a_latent


LTX2Runner.run_upsampler = run_upsampler_direct_decode

from lightx2v.infer import main  # noqa: E402

main()
PY
}

run_ltx2_direct_decode() {
  check_dir "${LIGHTX2V_REPO}"
  check_file "${LTX2_CONFIG_JSON}"

  local wrapper="${OUT_DIR}/tmp/run_ltx2_upsample_direct_decode.py"
  local output="${OUT_DIR}/videos/ltx2_upsample_direct_decode_seed${SEED}.mp4"
  write_ltx2_direct_wrapper "${wrapper}"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${output}" ]]; then
    echo "[ltx2] skip existing: ${output}"
    return
  fi

  local lightx2v_path="${LIGHTX2V_REPO}"
  local model_path="${LTX2_MODEL_PATH}"
  export lightx2v_path model_path
  # shellcheck source=/dev/null
  source "${LIGHTX2V_REPO}/scripts/base/base.sh"
  export DTYPE="${REQUESTED_DTYPE}"

  echo "[ltx2] 8-step t2av -> latent upsampler -> direct VAE decode"
  echo "[ltx2] output=${output}"
  python "${wrapper}" \
    --seed "${SEED}" \
    --model_cls ltx2 \
    --task t2av \
    --model_path "${LTX2_MODEL_PATH}" \
    --config_json "${LTX2_CONFIG_JSON}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${LTX2_NEGATIVE_PROMPT}" \
    --save_result_path "${output}" \
    --target_video_length "${LTX2_NUM_FRAMES}"
}

echo "Prompt: ${PROMPT}"
echo "Output root: ${OUT_DIR}"

if [[ "${RUN_DISTILL}" == "1" ]]; then
  run_distill_direct_decode
fi

if [[ "${RUN_LTX2}" == "1" ]]; then
  run_ltx2_direct_decode
fi

echo "Done. Videos are under: ${OUT_DIR}/videos"
