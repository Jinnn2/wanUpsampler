#!/usr/bin/env bash
set -euo pipefail

# 480x832 -> 720x1248 three-way Stage2 comparison on a shared prompt/seed set:
#   1. lora45_stage2:    step 45 LoRA x_pred -> Stage2 -> HR steps 46..50.
#   2. xpred45_stage2:   base step 45 x_pred -> Stage2 -> HR steps 46..50.
#   3. teacher50_stage2: base LR teacher at step 50 -> Stage2 -> decode.
#
# The third case intentionally performs the Stage2 handoff at the final step,
# so it is a clean-LR teacher + the same Stage2 operator reference rather than
# a direct 720p Wan sample.

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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Reuse the established 10-prompt evaluation set, not the generic 720p list.
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_stage2_three_way_480p}"
SEED="${SEED:-9700}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

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

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
done
for path in "${LORA_CKPT}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}" "${PROMPTS_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos" "${OUT_ROOT}/compare"
mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

export PROJECT_ROOT LORA_CKPT LORA_STRENGTH STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG
export TRAIN_STEP INFER_STEPS LR_H LR_W HR_H HR_W NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT
export STAGE2_USE_EMA STAGE2_RESIDUAL_SKIP

write_config() {
  local output="$1"
  local case_name="$2"
  python - "${output}" "${case_name}" <<'PY'
import json
import os
import sys

path, case_name = sys.argv[1:]
train_step = int(os.environ["TRAIN_STEP"])
infer_steps = int(os.environ["INFER_STEPS"])
residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
    raise SystemExit("STAGE2_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")

if case_name in {"lora45_stage2", "xpred45_stage2"}:
    change_step = train_step
elif case_name == "teacher50_stage2":
    change_step = infer_steps
else:
    raise SystemExit(f"Unsupported case: {case_name}")

config = {
    "infer_steps": infer_steps,
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
    "compare_name": case_name,
    "changing_resolution": True,
    "resolution_rate": [int(os.environ["LR_H"]) / int(os.environ["HR_H"])],
    "changing_resolution_steps": [change_step],
    "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
    "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
    "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
    "wan_clean_resizer_model_class": "stage2",
    "wan_clean_resizer_use_ema": os.environ["STAGE2_USE_EMA"] == "1",
}
if residual_skip != "checkpoint":
    config["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

if case_name == "lora45_stage2":
    config.update(
        {
            "lora_dynamic_apply": True,
            "lora_active_steps": [train_step],
            "lora_configs": [
                {
                    "name": "wan2.1",
                    "path": os.environ["LORA_CKPT"],
                    "strength": float(os.environ["LORA_STRENGTH"]),
                }
            ],
        }
    )

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

for case_name in lora45_stage2 xpred45_stage2 teacher50_stage2; do
  write_config "${OUT_ROOT}/configs/${case_name}.json" "${case_name}"
done

echo "[480p three-way] prompts=${PROMPTS_FILE} selected=${#prompts[@]} seed=${SEED}"
echo "[480p three-way] step=${TRAIN_STEP}, lora=${LORA_CKPT}, stage2=${STAGE2_CHECKPOINT}"
echo "[480p three-way] columns: LoRA@45 + Stage2 | x_pred@45 + Stage2 | teacher@50 + Stage2"

if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; configs written under ${OUT_ROOT}/configs and inference was not started."
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

run_infer() {
  local model_cls="$1"
  local case_name="$2"
  local sample_index="$3"
  local prompt="$4"
  local sample_seed="$5"
  local output_dir="${OUT_ROOT}/videos/${case_name}"
  local output="${output_dir}/${case_name}_${sample_index}_seed${sample_seed}.mp4"

  mkdir -p "${output_dir}"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${output}" ]]; then
    echo "  [skip] ${case_name}: ${output}"
    return
  fi

  python "${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py" \
    --seed "${sample_seed}" \
    --model_cls "${model_cls}" \
    --task t2v \
    --model_path "${MODEL_ROOT}" \
    --config_json "${OUT_ROOT}/configs/${case_name}.json" \
    --prompt "${prompt}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${output}" \
    --target_video_length "${NUM_FRAMES}"
}

make_compare() {
  local sample_index="$1"
  local sample_seed="$2"
  local out="${OUT_ROOT}/compare/${sample_index}_seed${sample_seed}_lora45_xpred45_teacher50_stage2_hstack.mp4"
  local v0="${OUT_ROOT}/videos/lora45_stage2/lora45_stage2_${sample_index}_seed${sample_seed}.mp4"
  local v1="${OUT_ROOT}/videos/xpred45_stage2/xpred45_stage2_${sample_index}_seed${sample_seed}.mp4"
  local v2="${OUT_ROOT}/videos/teacher50_stage2/teacher50_stage2_${sample_index}_seed${sample_seed}.mp4"

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[compare] ffmpeg unavailable; individual videos were still written."
    return
  fi
  if [[ -f "${v0}" && -f "${v1}" && -f "${v2}" ]]; then
    ffmpeg -hide_banner -loglevel error -y \
      -i "${v0}" -i "${v1}" -i "${v2}" \
      -filter_complex "[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];[2:v]setpts=PTS-STARTPTS[v2];[v0][v1][v2]hstack=inputs=3[v]" \
      -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${out}"
  else
    echo "[compare] skip hstack for index=${sample_index}; missing one or more case videos." >&2
  fi
}

index=0
for prompt in "${prompts[@]}"; do
  sample_index=$((PROMPT_OFFSET + index))
  sample_seed=$((SEED + sample_index))
  sample_label="$(printf "%02d" "${sample_index}")"

  echo "[$((index + 1))/${#prompts[@]}] index=${sample_label} seed=${sample_seed}"
  echo "${prompt}"
  run_infer "wan2.1_tail_skip_lora_clean_resizer_bridge" "lora45_stage2" "${sample_label}" "${prompt}" "${sample_seed}"
  run_infer "wan2.1_clean_resizer_bridge" "xpred45_stage2" "${sample_label}" "${prompt}" "${sample_seed}"
  run_infer "wan2.1_clean_resizer_bridge" "teacher50_stage2" "${sample_label}" "${prompt}" "${sample_seed}"
  make_compare "${sample_label}" "${sample_seed}"
  index=$((index + 1))
done

echo "480p three-way comparison videos: ${OUT_ROOT}/compare"
