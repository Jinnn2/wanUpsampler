#!/usr/bin/env bash
set -euo pipefail

# TAA-free oracle collection for sample-specific in-trajectory handoff research.
# Formal candidates: 30, 35, and every step from 40 through 50.
#
# Modes:
#   check        validate the locked protocol and write the resolved config
#   branch       run one LR prefix per prompt/seed and fork all candidate HR tails
#   independent  rerun every candidate as an independent warm pipeline for timing

MODE="${1:-branch}"
if [[ "${MODE}" == "run" ]]; then
  MODE="branch"
elif [[ "${MODE}" == "time" ]]; then
  MODE="independent"
fi
if [[ "${MODE}" != "check" && "${MODE}" != "branch" && "${MODE}" != "independent" ]]; then
  echo "Usage: $0 [check|branch|independent]" >&2
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

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_STAGE2_368X640_720X1248_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_STAGE2_368X640_720X1248_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_taa_free_oracle_branch_360p}"

FORMAL_CHANGE_STEPS="30 35 40 41 42 43 44 45 46 47 48 49 50"
CHANGE_STEPS="${CHANGE_STEPS:-${FORMAL_CHANGE_STEPS}}"
STRICT_PROTOCOL="${STRICT_PROTOCOL:-1}"
INFER_STEPS="${INFER_STEPS:-50}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9700}"

LR_H=368
LR_W=640
HR_H=720
HR_W=1248
LR_LATENT_H=46
LR_LATENT_W=80
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
PRECISION="${PRECISION:-bf16}"
USE_EMA="${USE_EMA:-0}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SAVE_LATENTS="${SAVE_LATENTS:-1}"
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE:-fp16}"
INCLUDE_NATIVE_HR="${INCLUDE_NATIVE_HR:-1}"
INDEPENDENT_WARMUP="${INDEPENDENT_WARMUP:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

if (( INFER_STEPS != 50 )); then
  echo "The formal oracle protocol requires INFER_STEPS=50." >&2
  exit 2
fi
if [[ "${STRICT_PROTOCOL}" == "1" && "${CHANGE_STEPS}" != "${FORMAL_CHANGE_STEPS}" ]]; then
  echo "Strict protocol requires CHANGE_STEPS='${FORMAL_CHANGE_STEPS}'." >&2
  echo "Set STRICT_PROTOCOL=0 only for a smoke run." >&2
  exit 2
fi
if [[ "${STRICT_PROTOCOL}" != "0" && "${STRICT_PROTOCOL}" != "1" ]]; then
  echo "STRICT_PROTOCOL must be 0 or 1." >&2
  exit 2
fi
if [[ "${SKIP_EXISTING}" != "0" && "${SKIP_EXISTING}" != "1" ]]; then
  echo "SKIP_EXISTING must be 0 or 1." >&2
  exit 2
fi
if [[ "${SAVE_LATENTS}" != "0" && "${SAVE_LATENTS}" != "1" ]]; then
  echo "SAVE_LATENTS must be 0 or 1." >&2
  exit 2
fi
if [[ "${INCLUDE_NATIVE_HR}" != "0" && "${INCLUDE_NATIVE_HR}" != "1" ]]; then
  echo "INCLUDE_NATIVE_HR must be 0 or 1." >&2
  exit 2
fi
if [[ "${INDEPENDENT_WARMUP}" != "0" && "${INDEPENDENT_WARMUP}" != "1" ]]; then
  echo "INDEPENDENT_WARMUP must be 0 or 1." >&2
  exit 2
fi
if [[ "${USE_EMA}" != "0" && "${USE_EMA}" != "1" ]]; then
  echo "USE_EMA must be 0 or 1." >&2
  exit 2
fi
case "${LATENT_SAVE_DTYPE}" in
  fp16|bf16|fp32) ;;
  *) echo "LATENT_SAVE_DTYPE must be fp16, bf16, or fp32." >&2; exit 2 ;;
esac
case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *) echo "PRECISION must be bf16 or fp16." >&2; exit 2 ;;
esac

read -r -a steps <<< "${CHANGE_STEPS}"
if (( ${#steps[@]} == 0 )); then
  echo "No candidate steps provided." >&2
  exit 2
fi
previous=0
for step in "${steps[@]}"; do
  if [[ ! "${step}" =~ ^[0-9]+$ ]] || (( step < 1 || step > INFER_STEPS )); then
    echo "Invalid candidate step '${step}'; expected an integer in [1, ${INFER_STEPS}]." >&2
    exit 2
  fi
  if (( step <= previous )); then
    echo "Candidate steps must be unique and strictly increasing: ${CHANGE_STEPS}" >&2
    exit 2
  fi
  previous="${step}"
done

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  [[ -d "${path}" ]] || { echo "Directory not found: ${path}" >&2; exit 1; }
done
for path in "${PROMPTS_FILE}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}"; do
  [[ -f "${path}" ]] || { echo "File not found: ${path}" >&2; exit 1; }
done
if [[ "${MODE}" != "check" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for WAN video export." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/configs"
export PROJECT_ROOT STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG INFER_STEPS NUM_FRAMES
export GUIDE_SCALE SAMPLE_SHIFT USE_EMA STAGE2_RESIDUAL_SKIP
export LR_H LR_W HR_H HR_W LR_LATENT_H LR_LATENT_W

CONFIG_JSON="${OUT_ROOT}/configs/taa_free_oracle_stage2.json"
python - "${CONFIG_JSON}" "${steps[0]}" <<'PY'
import json
import os
import sys

path, first_step = sys.argv[1], int(sys.argv[2])
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
    "parallel": False,
    "changing_resolution": True,
    "resolution_rate": [int(os.environ["LR_H"]) / int(os.environ["HR_H"])],
    "wan_lowres_latent_size": [
        int(os.environ["LR_LATENT_H"]),
        int(os.environ["LR_LATENT_W"]),
    ],
    "changing_resolution_steps": [first_step],
    "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
    "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
    "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
    "wan_clean_resizer_model_class": "stage2",
    "wan_clean_resizer_use_ema": os.environ["USE_EMA"] == "1",
}

residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
valid = {"checkpoint", "on", "off", "true", "false", "1", "0"}
if residual_skip not in valid:
    raise SystemExit(
        "STAGE2_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0"
    )
if residual_skip != "checkpoint":
    config["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

for forbidden in ("lora_configs", "lora_active_steps", "lora_dynamic_apply"):
    if forbidden in config:
        raise SystemExit(f"TAA-free protocol forbids {forbidden}")

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

echo "[TAA-free oracle] mode=${MODE}"
echo "[TAA-free oracle] candidates=${CHANGE_STEPS}"
echo "[TAA-free oracle] LR=368x640 (46x80 latent), HR=720x1248"
echo "[TAA-free oracle] TAA/LoRA=disabled, Stage2=${STAGE2_CHECKPOINT}"
echo "[TAA-free oracle] prompts=${PROMPTS_FILE}, offset=${PROMPT_OFFSET}, limit=${LIMIT}, seed=${START_SEED}"
echo "[TAA-free oracle] output=${OUT_ROOT}"

if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; resolved config written to ${CONFIG_JSON}."
  exit 0
fi

export CUDA_VISIBLE_DEVICES LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

args=(
  --seed "${START_SEED}"
  --model_cls "wan2.1_clean_resizer_bridge"
  --task t2v
  --model_path "${MODEL_ROOT}"
  --config_json "${CONFIG_JSON}"
  --prompts_file "${PROMPTS_FILE}"
  --prompt-offset "${PROMPT_OFFSET}"
  --limit "${LIMIT}"
  --change-steps "${CHANGE_STEPS}"
  --infer-steps "${INFER_STEPS}"
  --lr-height "${LR_H}"
  --lr-width "${LR_W}"
  --hr-height "${HR_H}"
  --hr-width "${HR_W}"
  --lr-latent-height "${LR_LATENT_H}"
  --lr-latent-width "${LR_LATENT_W}"
  --target_video_length "${NUM_FRAMES}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --out-root "${OUT_ROOT}"
  --execution-mode "${MODE}"
  --latent-save-dtype "${LATENT_SAVE_DTYPE}"
)
if [[ "${SAVE_LATENTS}" == "1" ]]; then
  args+=(--save-latents)
fi
if [[ "${INCLUDE_NATIVE_HR}" == "1" ]]; then
  args+=(--include-native-hr)
fi
if [[ "${MODE}" == "independent" && "${INDEPENDENT_WARMUP}" == "1" ]]; then
  args+=(--independent-warmup)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip-existing)
fi
if [[ "${STRICT_PROTOCOL}" == "1" ]]; then
  args+=(--strict-protocol)
else
  args+=(--no-strict-protocol)
fi

python "${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_oracle_branch_infer.py" \
  "${args[@]}"

if [[ "${MODE}" == "branch" ]]; then
  echo "TAA-free oracle videos: ${OUT_ROOT}/videos/step*/"
  echo "Router-state latents: ${OUT_ROOT}/latents/step*/"
  echo "Per-sample manifests: ${OUT_ROOT}/manifests/"
else
  echo "Independent timing videos/manifests: ${OUT_ROOT}/independent/"
fi
