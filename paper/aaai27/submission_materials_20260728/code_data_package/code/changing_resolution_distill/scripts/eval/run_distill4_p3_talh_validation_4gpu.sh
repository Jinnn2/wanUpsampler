#!/usr/bin/env bash
set -euo pipefail

# P3: 4 LoRA strengths x 2 renoise modes on eight validation-only prompts.
# Usage: all (default), check, prepare, generate, evaluate, or select.

ACTION="${1:-all}"
case "${ACTION}" in
  all|check|prepare|generate|evaluate|select) ;;
  *) echo "Usage: $0 [all|check|prepare|generate|evaluate|select]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/envs/vbench/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-/path/to/VBench}"
MODEL_ROOT="${DISTILL_MODEL_ROOT:-/path/to/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
DIT_CKPT="${DISTILL_DIT_CKPT:-${MODEL_ROOT}/distill_model.pt}"
STAGE2_CHECKPOINT="${DISTILL_STAGE2_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb/latest.pt}"
STAGE2_TRAIN_CONFIG="${DISTILL_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml}"
LORA_CHECKPOINT="${DISTILL_LORA_480_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/step_0010000.safetensors}"
PROMPTS_FILE="${TALH_VALIDATION_PROMPTS:-${PROJECT_ROOT}/paper/aaai27/experiments/distill4_talh_validation_prompts_8.txt}"
OUT_ROOT="${TALH_VALIDATION_ROOT:-${PROJECT_ROOT}/outputs/aaai27_experiments/distill4_talh_validation_sweep}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 4 )); then
  echo "Exactly four comma-separated GPU ids are required; got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

common=(
  --out-root "${OUT_ROOT}"
  --prompts "${PROMPTS_FILE}"
  --model-root "${MODEL_ROOT}"
  --dit-ckpt "${DIT_CKPT}"
  --stage2-checkpoint "${STAGE2_CHECKPOINT}"
  --stage2-train-config "${STAGE2_TRAIN_CONFIG}"
  --lora-checkpoint "${LORA_CHECKPOINT}"
  --strengths 0.25 0.5 0.75 1.0
  --renoise-modes random resize_flow
  --gpus "${GPUS[@]}"
  --seed "${SEED:-16000}"
  --limit "${LIMIT:-8}"
  --num-frames "${NUM_FRAMES:-81}"
  --guide-scale "${GUIDE_SCALE:-6.0}"
  --python "${WAN_PYTHON}"
  --stage2-use-ema
  --skip-existing
)

run_generate() {
  "${WAN_PYTHON}" \
    "${PROJECT_ROOT}/paper/aaai27/experiments/run_distill4_talh_validation_sweep.py" \
    run "${common[@]}"
}

run_evaluate() {
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VBENCH_PYTHON}" \
    "${PROJECT_ROOT}/paper/aaai27/experiments/run_vbench_factorials.py" run \
    --factorial-root "${OUT_ROOT}" \
    --vbench-root "${VBENCH_ROOT}" \
    --python "${VBENCH_PYTHON}" \
    --ngpus 4 \
    --dimension subject_consistency \
    --dimension background_consistency \
    --dimension motion_smoothness \
    --dimension aesthetic_quality \
    --dimension imaging_quality \
    --dimension temporal_flickering
}

run_select() {
  "${WAN_PYTHON}" \
    "${PROJECT_ROOT}/paper/aaai27/experiments/run_distill4_talh_validation_sweep.py" \
    select --out-root "${OUT_ROOT}"
}

case "${ACTION}" in
  check|prepare)
    "${WAN_PYTHON}" \
      "${PROJECT_ROOT}/paper/aaai27/experiments/run_distill4_talh_validation_sweep.py" \
      "${ACTION}" "${common[@]}"
    ;;
  generate) run_generate ;;
  evaluate) run_evaluate ;;
  select) run_select ;;
  all)
    run_generate
    run_evaluate
    run_select
    ;;
esac
