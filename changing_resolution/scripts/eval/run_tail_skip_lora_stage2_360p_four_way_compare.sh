#!/usr/bin/env bash
set -euo pipefail

# 10-prompt, four-column 368x640 -> 720x1248 comparison:
#   1. ori45 + Stage2
#   2. LoRA45 (strength=0.75) + Stage2
#   3. teacher50 + trilinear interpolation
#   4. teacher50 + Stage2
#
# 368x640 is the valid Wan "360p" working size: its latent is 46x80 (both
# even). Stage2 pixel-shuffles to 92x160 then center-crops to the 720x1248
# target latent size 90x156.

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

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
TRAIN_STEP="${TRAIN_STEP:-45}"
INFER_STEPS="${INFER_STEPS:-50}"
TAIL_SKIP_LORA_OUT_DIR="${TAIL_SKIP_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step${TRAIN_STEP}_to_step50}"
LORA_CKPT="${LORA_CKPT:-${TAIL_SKIP_LORA_OUT_DIR}/latest.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-0.75}"

STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_STAGE2_368X640_720X1248_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_STAGE2_368X640_720X1248_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml}}"
STAGE2_USE_EMA="${STAGE2_USE_EMA:-0}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"

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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_stage2_four_way_360p}"
SEED="${SEED:-9700}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

if (( TRAIN_STEP < 1 || TRAIN_STEP >= INFER_STEPS )); then
  echo "Invalid TRAIN_STEP=${TRAIN_STEP}; expected [1, $((INFER_STEPS - 1))]." >&2
  exit 2
fi
if [[ "${LORA_STRENGTH}" != "0.75" ]]; then
  echo "Warning: the selected 360p strength is 0.75; current override is ${LORA_STRENGTH}." >&2
fi

case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *) echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16." >&2; exit 2 ;;
esac

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  [[ -d "${path}" ]] || { echo "Directory not found: ${path}" >&2; exit 1; }
done
for path in "${LORA_CKPT}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}" "${PROMPTS_FILE}"; do
  [[ -f "${path}" ]] || { echo "File not found: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos" "${OUT_ROOT}/compare"
mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

export PROJECT_ROOT LORA_CKPT LORA_STRENGTH STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG
export TRAIN_STEP INFER_STEPS LR_H LR_W HR_H HR_W LR_LATENT_H LR_LATENT_W
export NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT STAGE2_USE_EMA STAGE2_RESIDUAL_SKIP

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
stage2_cases = {"ori45_stage2", "lora45_stage2", "teacher50_stage2"}
valid_cases = stage2_cases | {"teacher50_interp"}
if case_name not in valid_cases:
    raise SystemExit(f"Unsupported case: {case_name}")

change_step = train_step if case_name in {"ori45_stage2", "lora45_stage2"} else infer_steps
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
    "wan_lowres_latent_size": [
        int(os.environ["LR_LATENT_H"]),
        int(os.environ["LR_LATENT_W"]),
    ],
    "changing_resolution_steps": [change_step],
}

if case_name in stage2_cases:
    config.update(
        {
            "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
            "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
            "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
            "wan_clean_resizer_model_class": "stage2",
            "wan_clean_resizer_use_ema": os.environ["STAGE2_USE_EMA"] == "1",
        }
    )
    residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
    if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
        raise SystemExit("STAGE2_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")
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

for case_name in ori45_stage2 lora45_stage2 teacher50_interp teacher50_stage2; do
  write_config "${OUT_ROOT}/configs/${case_name}.json" "${case_name}"
done

echo "[360p four-way] prompts=${PROMPTS_FILE} selected=${#prompts[@]} seed=${SEED}"
echo "[360p four-way] LR=368x640 (latent 46x80), HR=720x1248 (latent 90x156)"
echo "[360p four-way] LoRA strength=${LORA_STRENGTH}, Stage2=${STAGE2_CHECKPOINT}"
echo "[360p four-way] columns: ori45+Stage2 | LoRA45+Stage2 | teacher50+interp | teacher50+Stage2"

if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; configs written under ${OUT_ROOT}/configs and inference was not started."
  exit 0
fi

export CUDA_VISIBLE_DEVICES LIGHTX2V_REPO
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
  local out="${OUT_ROOT}/compare/${sample_index}_seed${sample_seed}_four_way_hstack.mp4"
  local v0="${OUT_ROOT}/videos/ori45_stage2/ori45_stage2_${sample_index}_seed${sample_seed}.mp4"
  local v1="${OUT_ROOT}/videos/lora45_stage2/lora45_stage2_${sample_index}_seed${sample_seed}.mp4"
  local v2="${OUT_ROOT}/videos/teacher50_interp/teacher50_interp_${sample_index}_seed${sample_seed}.mp4"
  local v3="${OUT_ROOT}/videos/teacher50_stage2/teacher50_stage2_${sample_index}_seed${sample_seed}.mp4"

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[compare] ffmpeg unavailable; individual videos were still written."
    return
  fi
  if [[ -f "${v0}" && -f "${v1}" && -f "${v2}" && -f "${v3}" ]]; then
    ffmpeg -hide_banner -loglevel error -y \
      -i "${v0}" -i "${v1}" -i "${v2}" -i "${v3}" \
      -filter_complex "[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];[2:v]setpts=PTS-STARTPTS[v2];[3:v]setpts=PTS-STARTPTS[v3];[v0][v1][v2][v3]hstack=inputs=4[v]" \
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
  run_infer "wan2.1_clean_resizer_bridge" "ori45_stage2" "${sample_label}" "${prompt}" "${sample_seed}"
  run_infer "wan2.1_tail_skip_lora_clean_resizer_bridge" "lora45_stage2" "${sample_label}" "${prompt}" "${sample_seed}"
  run_infer "wan2.1_clean_interp_bridge" "teacher50_interp" "${sample_label}" "${prompt}" "${sample_seed}"
  run_infer "wan2.1_clean_resizer_bridge" "teacher50_stage2" "${sample_label}" "${prompt}" "${sample_seed}"
  make_compare "${sample_label}" "${sample_seed}"
  index=$((index + 1))
done

echo "360p four-way comparison videos: ${OUT_ROOT}/compare"
