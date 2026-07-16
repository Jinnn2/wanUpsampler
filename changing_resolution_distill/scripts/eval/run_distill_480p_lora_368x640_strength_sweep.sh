#!/usr/bin/env bash
set -euo pipefail

# Mirror the non-distill 368x640 strength-sweep protocol:
#   original3 -> 480p-trained LoRA strengths evaluated at 368x640 -> teacher4
# Stage2 is intentionally absent. This isolates cross-resolution trajectory
# correction before testing interaction with the spatial handoff operator.

MODE="${1:-run}"
if [[ "${MODE}" != "run" && "${MODE}" != "check" ]]; then
  echo "Usage: $0 [run|check]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

LORA_CKPT="${LORA_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/step_0010000.safetensors}"
CHECKPOINT_TAG="$(basename "${LORA_CKPT}" .safetensors)"
STRENGTHS="${STRENGTHS:-0.5 0.75 1.0}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_480p_lora_strength_sweep_360p_368x640_${CHECKPOINT_TAG}}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_368x640.json}"
LIMIT="${LIMIT:-10}"
SEED="${SEED:-9800}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
METRICS="${METRICS:-l1 mse psnr temporal_l1}"
PRIMARY_METRIC="${PRIMARY_METRIC:-psnr}"

for path in "${LORA_CKPT}" "${PROMPTS_FILE}" "${CONFIG_TEMPLATE}"; do
  [[ -f "${path}" ]] || { echo "File not found: ${path}" >&2; exit 1; }
done

echo "[distill 368p strength sweep] protocol=original3 -> LoRA strengths -> teacher4"
echo "  checkpoint : ${LORA_CKPT}"
echo "  resolution : 368x640"
echo "  strengths  : ${STRENGTHS}"
echo "  prompts    : ${PROMPTS_FILE} (limit=${LIMIT}, seed=${SEED})"
echo "  out_root   : ${OUT_ROOT}"
echo "  Stage2     : disabled"

if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; inference was not started."
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
LORA_CKPT="${LORA_CKPT}" \
HEIGHT=368 WIDTH=640 \
CONFIG_TEMPLATE="${CONFIG_TEMPLATE}" \
PROMPTS_FILE="${PROMPTS_FILE}" \
OUT_ROOT="${OUT_ROOT}" \
STRENGTHS="${STRENGTHS}" \
LIMIT="${LIMIT}" SEED="${SEED}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
METRICS="${METRICS}" PRIMARY_METRIC="${PRIMARY_METRIC}" \
bash "${SCRIPT_DIR}/run_last_step_skip_lora_strength_selection_480p.sh"

mkdir -p "${OUT_ROOT}/compare"
if command -v ffmpeg >/dev/null 2>&1; then
  read -r -a strength_values <<< "${STRENGTHS}"
  for ((i = 0; i < LIMIT; i++)); do
    label="$(printf "%02d" "${i}")"
    sample_seed=$((SEED + i))
    original="${OUT_ROOT}/_shared_baselines/videos/original3_clean_pred/original3_clean_pred_${label}_seed${sample_seed}.mp4"
    inputs=(
      -i "${original}"
    )
    filters=("[0:v]setpts=PTS-STARTPTS[v0]")
    stack_inputs="[v0]"
    input_index=1
    missing=0
    [[ -f "${original}" ]] || missing=1
    for strength in "${strength_values[@]}"; do
      tag="${strength//./p}"
      tag="${tag//-/m}"
      video="${OUT_ROOT}/strength_${tag}/videos/lora3_step3_clean_pred/lora3_step3_clean_pred_${label}_seed${sample_seed}.mp4"
      if [[ ! -f "${video}" ]]; then
        missing=1
        break
      fi
      inputs+=(-i "${video}")
      filters+=("[${input_index}:v]setpts=PTS-STARTPTS[v${input_index}]")
      stack_inputs+="[v${input_index}]"
      input_index=$((input_index + 1))
    done
    teacher="${OUT_ROOT}/_shared_baselines/videos/teacher4/teacher4_${label}_seed${sample_seed}.mp4"
    if [[ ! -f "${teacher}" ]]; then
      missing=1
    fi
    if [[ "${missing}" == "1" ]]; then
      echo "[compare] skip index=${label}; one or more videos are missing" >&2
      continue
    fi
    inputs+=(-i "${teacher}")
    filters+=("[${input_index}:v]setpts=PTS-STARTPTS[v${input_index}]")
    stack_inputs+="[v${input_index}]"
    filter_prefix="$(IFS=';'; echo "${filters[*]}")"
    output="${OUT_ROOT}/compare/${label}_seed${sample_seed}_strength_sweep_hstack.mp4"
    ffmpeg -hide_banner -loglevel error -y \
      "${inputs[@]}" \
      -filter_complex "${filter_prefix};${stack_inputs}hstack=inputs=$((input_index + 1))[v]" \
      -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${output}"
  done
else
  echo "[compare] ffmpeg unavailable; individual videos and metrics are complete."
fi

echo "Distill 368p strength summary: ${OUT_ROOT}/strength_metric_summary.csv"
echo "Distill 368p strength comparisons: ${OUT_ROOT}/compare"
