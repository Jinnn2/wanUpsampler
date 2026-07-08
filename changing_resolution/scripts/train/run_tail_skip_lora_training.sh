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
MODEL_ROOT="${MODEL_ROOT:-${JIN_ROOT}/Wan-AI/Wan2.1-T2V-1.3B}"
MODEL_CKPT="${MODEL_CKPT:-${MODEL_ROOT}/diffusion_pytorch_model.safetensors}"
TEXT_ENCODER_CKPT="${TEXT_ENCODER_CKPT:-${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"

TRAIN_STEP="${TRAIN_STEP:-45}"
CONFIG="${TAIL_SKIP_LORA_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_tail_skip_lora_step${TRAIN_STEP}.yaml}"
LMDB_DIR="${TAIL_SKIP_LORA_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_tail_skip_lora_step${TRAIN_STEP}_to_step50}"
OUT_DIR="${TAIL_SKIP_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step${TRAIN_STEP}_to_step50}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
DIST_BACKEND="${DIST_BACKEND:-}"
USER_MAX_STEPS="${MAX_STEPS:-}"
USER_MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_STEPS="${USER_MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USER_GRAD_ACCUM="${GRAD_ACCUM:-}"
BASE_GRAD_ACCUM="${BASE_GRAD_ACCUM:-8}"
if [[ -n "${USER_GRAD_ACCUM}" ]]; then
  GRAD_ACCUM="${USER_GRAD_ACCUM}"
elif (( NUM_GPUS > 1 )); then
  if (( BASE_GRAD_ACCUM < NUM_GPUS || BASE_GRAD_ACCUM % NUM_GPUS != 0 )); then
    echo "Cannot keep original effective batch with BASE_GRAD_ACCUM=${BASE_GRAD_ACCUM} and NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
  fi
  GRAD_ACCUM="$((BASE_GRAD_ACCUM / NUM_GPUS))"
else
  GRAD_ACCUM="${BASE_GRAD_ACCUM}"
fi
LR="${LR:-5e-5}"
PRECISION="${PRECISION:-bf16}"
MAX_SAMPLES="${USER_MAX_SAMPLES}"
RESUME="${RESUME:-}"
MODEL_PATHS="${MODEL_PATHS:-[\"${MODEL_CKPT}\",\"${TEXT_ENCODER_CKPT}\"]}"
MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
LORA_RANK="${LORA_RANK:-}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"
ENABLE_CFG="${ENABLE_CFG:-1}"

export CUDA_VISIBLE_DEVICES
export NUM_GPUS
if [[ -n "${DIST_BACKEND}" ]]; then
  export DIST_BACKEND
fi
export DIFFSYNTH_REPO
export LIGHTX2V_REPO
export PYTHONPATH="${DIFFSYNTH_REPO}:${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

MODE="${1:-train}"

check_env() {
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
  fi
  if [[ ! -d "${LMDB_DIR}" ]] || [[ -z "$(find "${LMDB_DIR}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
    echo "Tail-skip LoRA LMDB not found: ${LMDB_DIR}" >&2
    echo "Build it first with: TRAIN_STEP=${TRAIN_STEP} bash changing_resolution/scripts/data/build_tail_skip_lora_lmdb.sh" >&2
    exit 1
  fi
  if [[ ! -f "${DIFFSYNTH_REPO}/examples/wanvideo/model_training/train.py" ]]; then
    echo "DiffSynth train.py not found. Set DIFFSYNTH_REPO." >&2
    exit 1
  fi
  if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "Wan DiT checkpoint not found: ${MODEL_CKPT}" >&2
    exit 1
  fi
  if [[ ! -f "${TEXT_ENCODER_CKPT}" ]]; then
    echo "Text encoder checkpoint not found: ${TEXT_ENCODER_CKPT}" >&2
    exit 1
  fi
  if [[ -n "${LORA_CHECKPOINT}" && ! -f "${LORA_CHECKPOINT}" ]]; then
    echo "LoRA checkpoint not found: ${LORA_CHECKPOINT}" >&2
    exit 1
  fi
  if (( NUM_GPUS > 1 )) && ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun not found. Install PyTorch torchrun or set NUM_GPUS=1." >&2
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
    raise SystemExit("Missing python packages: " + ", ".join(missing))
PY
}

train_lora() {
  check_env
  mkdir -p "${OUT_DIR}"

  args=(
    --config "${CONFIG}"
    --data_dir "${LMDB_DIR}"
    --out_dir "${OUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --max_steps "${MAX_STEPS}"
    --precision "${PRECISION}"
    --train_step "${TRAIN_STEP}"
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
  if [[ -n "${LORA_CHECKPOINT}" ]]; then
    args+=(--lora_checkpoint "${LORA_CHECKPOINT}")
  fi
  if [[ "${ENABLE_CFG}" == "1" ]]; then
    args+=(--enable_cfg)
  else
    args+=(--no-enable_cfg)
  fi

  echo "Wan2.1 tail-skip LoRA training"
  echo "  project     : ${PROJECT_ROOT}"
  echo "  config      : ${CONFIG}"
  echo "  data        : ${LMDB_DIR}"
  echo "  out_dir     : ${OUT_DIR}"
  echo "  diffsynth   : ${DIFFSYNTH_REPO}"
  echo "  model       : ${MODEL_ROOT}"
  echo "  train_step  : ${TRAIN_STEP}"
  echo "  num_gpus    : ${NUM_GPUS}"
  echo "  grad_accum  : ${GRAD_ACCUM}"
  echo "  eff_batch   : $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
  echo "  steps       : ${MAX_STEPS}"
  echo "  enable_cfg  : ${ENABLE_CFG}"

  if (( NUM_GPUS > 1 )); then
    torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
      "${PROJECT_ROOT}/changing_resolution/scripts/train/train_tail_skip_lora.py" "${args[@]}"
  else
    python "${PROJECT_ROOT}/changing_resolution/scripts/train/train_tail_skip_lora.py" "${args[@]}"
  fi
}

case "${MODE}" in
  check)
    check_env
    echo "Tail-skip LoRA training preflight passed."
    ;;
  train)
    train_lora
    ;;
  smoke)
    MAX_STEPS="${USER_MAX_STEPS:-200}" MAX_SAMPLES="${USER_MAX_SAMPLES:-64}" train_lora
    ;;
  *)
    echo "Usage: bash changing_resolution/scripts/train/run_tail_skip_lora_training.sh [check|smoke|train]" >&2
    exit 2
    ;;
esac
