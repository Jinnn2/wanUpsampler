#!/usr/bin/env bash
set -euo pipefail

# Evaluate whether a 50-step tail-skip LoRA transfers to the 360p class
# resolution 368x640. Wan requires even spatial latent dimensions: 360x624
# would encode to 45x78 and the DiT patch path crops the odd height to 44.
# Every case uses the same prompt and seed:
#   ori_45  : base Wan x0 prediction after the step-45 denoise, then stop.
#   lora_45 : LoRA enabled only for the step-45 denoise, then stop.
#   ori_50  : base Wan full 50-step result; metric reference for usefulness.
#
# `ori_45` versus `lora_45` shows whether the LoRA has an effect.  `ori_50`
# is required to determine whether that effect moves the 45-step prediction
# closer to the original model's fully denoised output.

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

HEIGHT="${HEIGHT:-368}"
WIDTH="${WIDTH:-640}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
PRECISION="${PRECISION:-bf16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_clean_pred_compare_360p_368x640}"
SEED="${SEED:-9700}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

RUN_METRICS="${RUN_METRICS:-1}"
METRICS="${METRICS:-l1 mse psnr temporal_l1}"
METRICS_CPU="${METRICS_CPU:-0}"

if [[ "${HEIGHT}" != "368" || "${WIDTH}" != "640" ]]; then
  echo "This entrypoint is locked to the 360p-class test size 368x640." >&2
  echo "Received HEIGHT=${HEIGHT}, WIDTH=${WIDTH}." >&2
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
for path in "${LORA_CKPT}" "${PROMPTS_FILE}"; do
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

export PROJECT_ROOT LORA_CKPT LORA_STRENGTH TRAIN_STEP INFER_STEPS HEIGHT WIDTH NUM_FRAMES
export GUIDE_SCALE SAMPLE_SHIFT

write_config() {
  local output="$1"
  local case_name="$2"
  python - "${output}" "${case_name}" <<'PY'
import json
import os
import sys

path, case_name = sys.argv[1:]
train_step = int(os.environ["TRAIN_STEP"])
config = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": int(os.environ["HEIGHT"]),
    "target_width": int(os.environ["WIDTH"]),
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": float(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "compare_name": case_name,
}

if case_name in {"ori_45", "lora_45"}:
    config.update(
        {
            "lora_dynamic_apply": True,
            "lora_active_steps": [train_step],
            "return_clean_pred_steps": [train_step],
            "lora_configs": [
                {
                    "name": "wan2.1",
                    "path": os.environ["LORA_CKPT"],
                    "strength": 0.0 if case_name == "ori_45" else float(os.environ["LORA_STRENGTH"]),
                }
            ],
        }
    )
elif case_name != "ori_50":
    raise SystemExit(f"Unsupported case: {case_name}")

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

for case_name in ori_45 lora_45 ori_50; do
  write_config "${OUT_ROOT}/configs/${case_name}.json" "${case_name}"
done

echo "[360p LoRA eval] prompts=${PROMPTS_FILE} selected=${#prompts[@]} seed=${SEED}"
echo "[360p LoRA eval] resolution=${HEIGHT}x${WIDTH} (latent 46x80), step=${TRAIN_STEP}, lora=${LORA_CKPT}"
echo "[360p LoRA eval] columns: ori_45 | lora_45 | ori_50"

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
  local out="${OUT_ROOT}/compare/${sample_index}_seed${sample_seed}_ori45_lora45_ori50_hstack.mp4"
  local v0="${OUT_ROOT}/videos/ori_45/ori_45_${sample_index}_seed${sample_seed}.mp4"
  local v1="${OUT_ROOT}/videos/lora_45/lora_45_${sample_index}_seed${sample_seed}.mp4"
  local v2="${OUT_ROOT}/videos/ori_50/ori_50_${sample_index}_seed${sample_seed}.mp4"

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[compare] ffmpeg unavailable; individual videos were still written."
    return
  fi
  if [[ -f "${v0}" && -f "${v1}" && -f "${v2}" ]]; then
    ffmpeg -hide_banner -loglevel error -y \
      -i "${v0}" -i "${v1}" -i "${v2}" \
      -filter_complex "[0:v]scale=384:-2,setpts=PTS-STARTPTS[v0];[1:v]scale=384:-2,setpts=PTS-STARTPTS[v1];[2:v]scale=384:-2,setpts=PTS-STARTPTS[v2];[v0][v1][v2]hstack=inputs=3[v]" \
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
  run_infer "wan2.1_tail_skip_lora" "ori_45" "${sample_label}" "${prompt}" "${sample_seed}"
  run_infer "wan2.1_tail_skip_lora" "lora_45" "${sample_label}" "${prompt}" "${sample_seed}"
  run_infer "wan2.1" "ori_50" "${sample_label}" "${prompt}" "${sample_seed}"
  make_compare "${sample_label}" "${sample_seed}"
  index=$((index + 1))
done

if [[ "${RUN_METRICS}" == "1" ]]; then
  read -r -a metric_args <<<"${METRICS}"
  cpu_args=()
  if [[ "${METRICS_CPU}" == "1" ]]; then
    cpu_args+=(--cpu)
  fi
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/eval_last_step_skip_lora_video_closeness.py" \
    --out_root "${OUT_ROOT}" \
    --original_case ori_45 \
    --lora_case lora_45 \
    --teacher_case ori_50 \
    --metrics "${metric_args[@]}" \
    "${cpu_args[@]}"
fi

echo "360p LoRA comparison videos: ${OUT_ROOT}/compare"
echo "Metric summary (if enabled): ${OUT_ROOT}/metrics/original_lora_teacher_summary.csv"
