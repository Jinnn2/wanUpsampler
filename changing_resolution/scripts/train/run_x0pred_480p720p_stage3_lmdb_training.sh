#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
DENOISE_STEP="${DENOISE_STEP:-45}"
HR_TARGET_MODE="${HR_TARGET_MODE:-x0_pred}"
case "${HR_TARGET_MODE}" in
  x0_pred)
    DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_x0predhr_step${DENOISE_STEP}"
    DEFAULT_OUT_NAME="changing_resolution_x0pred_480p720p_stage3_x0predhr_step${DENOISE_STEP}_lmdb"
    ;;
  clean)
    DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_cleanhr_step${DENOISE_STEP}"
    DEFAULT_OUT_NAME="changing_resolution_x0pred_480p720p_stage3_cleanhr_step${DENOISE_STEP}_lmdb"
    ;;
  *)
    DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_${HR_TARGET_MODE}_step${DENOISE_STEP}"
    DEFAULT_OUT_NAME="changing_resolution_x0pred_480p720p_stage3_${HR_TARGET_MODE}_step${DENOISE_STEP}_lmdb"
    ;;
esac
LMDB_DIR="${CR_STAGE3_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/${DEFAULT_LMDB_NAME}}"
CONFIG="${CR_STAGE3_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_x0pred_480p_to_720p_lmdb_stage3.yaml}"
OUT_DIR="${CR_STAGE3_OUT_DIR:-${PROJECT_ROOT}/outputs/${DEFAULT_OUT_NAME}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
PRECISION="${PRECISION:-bf16}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-8}"
SCALE_FACTOR="${SCALE_FACTOR:-1.5}"
RESUME="${RESUME:-}"
NO_RESIDUAL_SKIP="${NO_RESIDUAL_SKIP:-true}"

export CUDA_VISIBLE_DEVICES
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

MODE="${1:-train}"

check_lmdb() {
  if [[ ! -d "${LMDB_DIR}" ]] || [[ -z "$(find "${LMDB_DIR}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
    echo "No Stage 3 x0-pred LMDB shards found under: ${LMDB_DIR}" >&2
    echo "Build it first with:" >&2
    echo "  DENOISE_STEP=${DENOISE_STEP} bash changing_resolution/scripts/data/build_x0pred_480p720p_stage3_lmdb.sh" >&2
    exit 1
  fi
}

check_python_deps() {
  python - <<'PY'
import importlib.util
missing = [name for name in ("lmdb", "torch", "yaml", "tqdm") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing python packages: " + ", ".join(missing) + ". Run: pip install -r requirements.txt")
PY
}

check_stage3_import() {
  python - <<'PY'
import torch
from wan_sr.data import X0PredLatentLMDBDataset
from wan_sr.models import WanCleanLatentResizerStage2
model = WanCleanLatentResizerStage2(hidden_channels=32, num_res_blocks=2)
x = torch.randn(1, 16, 2, 4, 6)
y = model(x, output_size=(6, 9))
if tuple(y.shape) != (1, 16, 2, 6, 9):
    raise SystemExit(f"Unexpected Stage 3 smoke-test output shape: {tuple(y.shape)}")
print(f"Stage 3 model/import smoke test passed: shape={tuple(y.shape)} dataset={X0PredLatentLMDBDataset.__name__}")
PY
}

train_stage3() {
  check_lmdb
  check_python_deps
  mkdir -p "${OUT_DIR}"

  local resume_args=()
  if [[ -n "${RESUME}" ]]; then
    resume_args=(--resume "${RESUME}")
  fi

  local residual_args=()
  if [[ "${NO_RESIDUAL_SKIP}" == "true" ]]; then
    residual_args=(--no_residual_skip)
  fi

  echo "Stage 3 x0-pred latent LMDB training"
  echo "  project : ${PROJECT_ROOT}"
  echo "  lmdb    : ${LMDB_DIR}"
  echo "  config  : ${CONFIG}"
  echo "  out_dir : ${OUT_DIR}"
  echo "  step    : ${DENOISE_STEP}"
  echo "  gpu     : ${CUDA_VISIBLE_DEVICES}"
  echo "  steps   : ${MAX_STEPS}"

  python "${PROJECT_ROOT}/changing_resolution/scripts/train/train_x0pred_latent_resizer_stage3.py" \
    --config "${CONFIG}" \
    --data_dir "${LMDB_DIR}" \
    --out_dir "${OUT_DIR}" \
    --denoise_step "${DENOISE_STEP}" \
    --hidden_channels "${HIDDEN_CHANNELS}" \
    --num_res_blocks "${NUM_RES_BLOCKS}" \
    --scale_factor "${SCALE_FACTOR}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max_steps "${MAX_STEPS}" \
    --precision "${PRECISION}" \
    "${residual_args[@]}" \
    "${resume_args[@]}"
}

case "${MODE}" in
  check)
    check_lmdb
    check_python_deps
    check_stage3_import
    echo "Stage 3 LMDB training preflight passed: ${LMDB_DIR} (denoise_step=${DENOISE_STEP})"
    ;;
  train)
    train_stage3
    ;;
  *)
    echo "Usage: bash changing_resolution/scripts/train/run_x0pred_480p720p_stage3_lmdb_training.sh [check|train]" >&2
    exit 2
    ;;
esac
