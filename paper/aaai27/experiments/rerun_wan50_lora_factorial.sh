#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

MODE="${1:-run}"
STEPS="${STEPS:-40 45}"
PROMPTS="${AAAI_PROMPTS:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${WAN50_FACTORIAL:-${PROJECT_ROOT}/outputs/aaai27_experiments/factorial_wan50}"
STAGE2_CKPT="${WAN50_STAGE2_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb/latest.pt}"
STAGE2_CONFIG="${WAN50_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml}"
LORA40_CKPT="${WAN50_LORA40_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step40_to_step50_temporal/latest.safetensors}"
LORA45_CKPT="${WAN50_LORA45_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step45_to_step50/latest.safetensors}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SEED="${SEED:-9700}"

read -r -a STEP_ARGS <<< "${STEPS}"
LORA_ARGS=()
for step in "${STEP_ARGS[@]}"; do
  case "${step}" in
    40) LORA_ARGS+=(--lora-checkpoint "40=${LORA40_CKPT}" --lora-strength-step "40=1.0") ;;
    45) LORA_ARGS+=(--lora-checkpoint "45=${LORA45_CKPT}") ;;
    *) echo "Unsupported Wan50 LoRA handoff step: ${step}" >&2; exit 2 ;;
  esac
done

echo "Rerun Wan50 LoRA factorial branches"
echo "  mode      : ${MODE}"
echo "  steps     : ${STEPS}"
echo "  out_root  : ${OUT_ROOT}"
echo "  lora40    : ${LORA40_CKPT}"
echo "  lora45    : ${LORA45_CKPT}"
echo "  overwrite : LoRA branches only; Base videos are untouched"

python paper/aaai27/experiments/run_factorial.py "${MODE}" \
  --family wan50 \
  --steps "${STEP_ARGS[@]}" \
  --handoffs lora \
  --resizers interp stage2 \
  --prompts "${PROMPTS}" \
  --out-root "${OUT_ROOT}" \
  --stage2-checkpoint "${STAGE2_CKPT}" \
  --stage2-train-config "${STAGE2_CONFIG}" \
  "${LORA_ARGS[@]}" \
  --lora-strength 0.75 \
  --limit "${LIMIT}" \
  --prompt-offset "${PROMPT_OFFSET}" \
  --seed "${SEED}" \
  --no-skip-existing \
  --manifest-name lora_rerun_manifest.json
