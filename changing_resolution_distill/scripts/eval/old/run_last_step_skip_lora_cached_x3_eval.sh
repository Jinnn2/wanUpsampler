#!/usr/bin/env bash
set -euo pipefail

# Cached-x3 closed-loop eval.
#
# DiffSynth backend:
#   z_orig3 / z_lora3 are computed through the DiffSynth training module.
#
# LightX2V backend:
#   z_orig3 / z_lora3 are computed through the LightX2V bridge/runtime.
#
# Usage:
#   bash changing_resolution_distill/scripts/eval/run_last_step_skip_lora_cached_x3_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

JIN_ROOT="${JIN_ROOT:-/mnt/afs_2/houze}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-${JIN_ROOT}/LightX2V}"
DIFFSYNTH_REPO="${DIFFSYNTH_REPO:-${JIN_ROOT}/DiffSynth-Studio}"
CR_DISTILL_MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-${JIN_ROOT}/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CR_DISTILL_DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${CR_DISTILL_MODEL_ROOT}/distill_model.pt}"
CR_DISTILL_TEXT_ENCODER_CKPT="${CR_DISTILL_TEXT_ENCODER_CKPT:-${CR_DISTILL_MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
CR_DISTILL_LORA_LMDB_DIR="${CR_DISTILL_LORA_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3}"
CR_DISTILL_LORA_PLAN_D_OUT_DIR="${CR_DISTILL_LORA_PLAN_D_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_plan_d_rank16_qkvo_ffn}"
LORA_CKPT="${LORA_CKPT:-${CR_DISTILL_LORA_PLAN_D_OUT_DIR}/latest.safetensors}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DTYPE="${DTYPE:-BF16}"
PRECISION="${PRECISION:-bf16}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
INDICES="${INDICES:-}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"
RUN_DIFFSYNTH="${RUN_DIFFSYNTH:-1}"
RUN_LIGHTX2V="${RUN_LIGHTX2V:-1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_cached_x3_eval}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_bridge_720p.example.json}"

if [[ ! -f "${LORA_CKPT}" ]]; then
  echo "LoRA checkpoint not found: ${LORA_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${CR_DISTILL_LORA_LMDB_DIR}" ]]; then
  echo "LoRA LMDB not found: ${CR_DISTILL_LORA_LMDB_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/configs"

export CUDA_VISIBLE_DEVICES
export DTYPE
export DIFFSYNTH_REPO
export LIGHTX2V_REPO
export PYTHONPATH="${DIFFSYNTH_REPO}:${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

BASE_CONFIG="${OUT_ROOT}/configs/cached_x3_lightx2v_base.json"
LORA_CONFIG="${OUT_ROOT}/configs/cached_x3_lightx2v_lora.json"

python - \
  "${CONFIG_TEMPLATE}" \
  "${BASE_CONFIG}" \
  "${LORA_CONFIG}" \
  "${CR_DISTILL_DIT_CKPT}" \
  "${LORA_CKPT}" \
  "${LORA_STRENGTH}" <<'PY'
import json
import sys
from pathlib import Path

src, base_dst, lora_dst, ckpt, lora_ckpt, lora_strength = sys.argv[1:]
template = json.loads(Path(src).read_text(encoding="utf-8"))
for key in list(template.keys()):
    if (
        key.startswith("wan_clean_resizer")
        or key in {
            "changing_resolution",
            "resolution_rate",
            "changing_resolution_steps",
            "wan_distill_bridge_renoise_mode",
        }
    ):
        template.pop(key, None)

common = {
    **template,
    "infer_steps": 4,
    "target_video_length": 81,
    "target_height": 720,
    "target_width": 1248,
    "sample_guide_scale": 6,
    "sample_shift": 5,
    "enable_cfg": False,
    "feature_caching": "NoCaching",
    "denoising_step_list": [1000, 750, 500, 250],
    "dit_original_ckpt": str(ckpt),
}
base = dict(common)
lora = dict(common)
lora.update({
    "lora_dynamic_apply": True,
    "lora_active_steps": [3],
    "lora_configs": [
        {
            "name": "wan2.1",
            "path": str(lora_ckpt),
            "strength": float(lora_strength),
        }
    ],
})
Path(base_dst).write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(lora_dst).write_text(json.dumps(lora, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

common_args=(
  --data_dir "${CR_DISTILL_LORA_LMDB_DIR}"
  --out_dir "${OUT_ROOT}"
  --lora_ckpt "${LORA_CKPT}"
  --num_samples "${NUM_SAMPLES}"
  --precision "${PRECISION}"
)
if [[ -n "${INDICES}" ]]; then
  common_args+=(--indices "${INDICES}")
fi

if [[ "${RUN_DIFFSYNTH}" == "1" ]]; then
  MODEL_PATHS="${MODEL_PATHS:-[\"${CR_DISTILL_DIT_CKPT}\",\"${CR_DISTILL_TEXT_ENCODER_CKPT}\"]}"
  echo "[cached-x3] DiffSynth backend"
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/eval_last_step_skip_lora_cached_x3_loss.py" \
    --backend diffsynth \
    "${common_args[@]}" \
    --model_paths "${MODEL_PATHS}" \
    --lora_rank 16 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
fi

if [[ "${RUN_LIGHTX2V}" == "1" ]]; then
  echo "[cached-x3] LightX2V backend"
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/eval_last_step_skip_lora_cached_x3_loss.py" \
    --backend lightx2v \
    "${common_args[@]}" \
    --model_path "${CR_DISTILL_MODEL_ROOT}" \
    --lightx2v_base_config "${BASE_CONFIG}" \
    --lightx2v_lora_config "${LORA_CONFIG}" \
    --lora_strength "${LORA_STRENGTH}"
fi

echo "[cached-x3] done: ${OUT_ROOT}"
