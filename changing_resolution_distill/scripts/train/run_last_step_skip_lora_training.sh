#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

JIN_ROOT="${JIN_ROOT:-/mnt/afs_2/houze}"
DIFFSYNTH_REPO="${DIFFSYNTH_REPO:-${JIN_ROOT}/DiffSynth-Studio}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-${JIN_ROOT}/LightX2V}"
CR_DISTILL_MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-${JIN_ROOT}/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CR_DISTILL_DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${CR_DISTILL_MODEL_ROOT}/distill_model.pt}"
CR_DISTILL_TEXT_ENCODER_CKPT="${CR_DISTILL_TEXT_ENCODER_CKPT:-${CR_DISTILL_MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
CR_DISTILL_LORA_CONFIG="${CR_DISTILL_LORA_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml}"
CR_DISTILL_LORA_LMDB_DIR="${CR_DISTILL_LORA_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3}"
CR_DISTILL_LORA_OUT_DIR="${CR_DISTILL_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USER_MAX_STEPS="${MAX_STEPS:-}"
USER_MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_STEPS="${USER_MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
PRECISION="${PRECISION:-bf16}"
MAX_SAMPLES="${USER_MAX_SAMPLES}"
RESUME="${RESUME:-}"
MODEL_PATHS="${MODEL_PATHS:-[\"${CR_DISTILL_DIT_CKPT}\",\"${CR_DISTILL_TEXT_ENCODER_CKPT}\"]}"
MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
LORA_RANK="${LORA_RANK:-}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
TRAINING_MODE="${TRAINING_MODE:-cached_x_pre_step3}"

export CUDA_VISIBLE_DEVICES
export DIFFSYNTH_REPO
export LIGHTX2V_REPO
export PYTHONPATH="${DIFFSYNTH_REPO}:${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

MODE="${1:-train}"

check_env() {
  if [[ ! -f "${CR_DISTILL_LORA_CONFIG}" ]]; then
    echo "Config not found: ${CR_DISTILL_LORA_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -d "${CR_DISTILL_LORA_LMDB_DIR}" ]] || [[ -z "$(find "${CR_DISTILL_LORA_LMDB_DIR}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
    echo "LoRA LMDB not found: ${CR_DISTILL_LORA_LMDB_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${DIFFSYNTH_REPO}/examples/wanvideo/model_training/train.py" ]]; then
    echo "DiffSynth train.py not found. Set DIFFSYNTH_REPO or run setup_last_step_skip_lora_env.sh install." >&2
    exit 1
  fi
  if [[ ! -f "${CR_DISTILL_DIT_CKPT}" ]]; then
    echo "DiT checkpoint not found: ${CR_DISTILL_DIT_CKPT}" >&2
    exit 1
  fi
  if [[ ! -f "${CR_DISTILL_TEXT_ENCODER_CKPT}" ]]; then
    echo "Text encoder checkpoint not found: ${CR_DISTILL_TEXT_ENCODER_CKPT}" >&2
    exit 1
  fi
  python - <<'PY'
import importlib.util
missing = [
    name
    for name in ("torch", "accelerate", "safetensors", "diffsynth", "modelscope", "lmdb", "yaml")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing python packages: "
        + ", ".join(missing)
        + "\nRun: bash changing_resolution_distill/scripts/train/setup_last_step_skip_lora_env.sh install"
    )
PY
}

train_lora() {
  check_env
  mkdir -p "${CR_DISTILL_LORA_OUT_DIR}"

  args=(
    --config "${CR_DISTILL_LORA_CONFIG}"
    --data_dir "${CR_DISTILL_LORA_LMDB_DIR}"
    --out_dir "${CR_DISTILL_LORA_OUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --max_steps "${MAX_STEPS}"
    --precision "${PRECISION}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    args+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${RESUME}" ]]; then
    args+=(--resume "${RESUME}")
  fi
  if [[ -n "${MODEL_PATHS}" ]]; then
    args+=(--model_paths "${MODEL_PATHS}")
  fi
  if [[ -n "${MODEL_ID_WITH_ORIGIN_PATHS}" ]]; then
    args+=(--model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}")
  fi
  if [[ -n "${TOKENIZER_PATH}" ]]; then
    args+=(--tokenizer_path "${TOKENIZER_PATH}")
  fi
  if [[ -n "${LORA_RANK}" ]]; then
    args+=(--lora_rank "${LORA_RANK}")
  fi
  if [[ -n "${LORA_TARGET_MODULES}" ]]; then
    args+=(--lora_target_modules "${LORA_TARGET_MODULES}")
  fi
  if [[ -n "${TRAINING_MODE}" ]]; then
    args+=(--training_mode "${TRAINING_MODE}")
  fi

  echo "Cached x_pre_step3 last-step-skip LoRA training"
  echo "  project     : ${PROJECT_ROOT}"
  echo "  config      : ${CR_DISTILL_LORA_CONFIG}"
  echo "  data        : ${CR_DISTILL_LORA_LMDB_DIR}"
  echo "  out_dir     : ${CR_DISTILL_LORA_OUT_DIR}"
  echo "  diffsynth   : ${DIFFSYNTH_REPO}"
  echo "  gpu         : ${CUDA_VISIBLE_DEVICES}"
  echo "  steps       : ${MAX_STEPS}"
  echo "  max_samples : ${MAX_SAMPLES:-all}"
  echo "  lora_rank   : ${LORA_RANK:-config}"
  echo "  lora_target : ${LORA_TARGET_MODULES:-config}"
  echo "  mode        : ${TRAINING_MODE}"

  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/train/train_last_step_skip_lora.py" "${args[@]}"
}

case "${MODE}" in
  check)
    check_env
    echo "Last-step-skip LoRA training preflight passed."
    ;;
  train)
    train_lora
    ;;
  smoke)
    MAX_STEPS="${USER_MAX_STEPS:-200}" MAX_SAMPLES="${USER_MAX_SAMPLES:-64}" train_lora
    ;;
  *)
    echo "Usage: bash changing_resolution_distill/scripts/train/run_last_step_skip_lora_training.sh [check|smoke|train]" >&2
    exit 2
    ;;
esac
