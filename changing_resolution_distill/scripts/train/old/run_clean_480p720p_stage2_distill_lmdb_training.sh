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
CR_DISTILL_STAGE2_TAG="${CR_DISTILL_STAGE2_TAG:-14b_cfgdistill_5k}"
LMDB_DIR="${CR_DISTILL_STAGE2_LMDB_DIR:-${CR_DISTILL_CLEAN_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_clean_480p720p_${CR_DISTILL_STAGE2_TAG}}}"
CONFIG="${CR_DISTILL_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml}"
OUT_DIR="${CR_DISTILL_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_480p720p_stage2_${CR_DISTILL_STAGE2_TAG}_lmdb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
EMA_DECAY="${EMA_DECAY:-}"
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
    echo "No distill clean-latent LMDB shards found under: ${LMDB_DIR}" >&2
    echo "Build it first with:" >&2
    echo "  TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 OVERWRITE_LMDB=1 bash changing_resolution_distill/scripts/data/build_clean_480p720p_14b_cfgdistill_lmdb_1k_multigpu.sh lmdb" >&2
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

check_stage2_import() {
  python - <<'PY'
import torch
from wan_sr.data import CleanLatentLMDBDataset
from wan_sr.models import WanCleanLatentResizerStage2
model = WanCleanLatentResizerStage2(hidden_channels=32, num_res_blocks=2)
x = torch.randn(1, 16, 2, 4, 6)
y = model(x, output_size=(6, 9))
if tuple(y.shape) != (1, 16, 2, 6, 9):
    raise SystemExit(f"Unexpected Stage 2 distill smoke-test output shape: {tuple(y.shape)}")
print(f"Stage 2 distill model/import smoke test passed: shape={tuple(y.shape)} dataset={CleanLatentLMDBDataset.__name__}")
PY
}

train_stage2_distill() {
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

  local ema_args=()
  if [[ -n "${EMA_DECAY}" ]]; then
    ema_args=(--ema_decay "${EMA_DECAY}")
  fi

  echo "Stage 2 distill clean latent LMDB training"
  echo "  project   : ${PROJECT_ROOT}"
  echo "  tag       : ${CR_DISTILL_STAGE2_TAG}"
  echo "  lmdb      : ${LMDB_DIR}"
  echo "  config    : ${CONFIG}"
  echo "  out_dir   : ${OUT_DIR}"
  echo "  gpu       : ${CUDA_VISIBLE_DEVICES}"
  echo "  steps     : ${MAX_STEPS}"
  echo "  ema_decay : ${EMA_DECAY:-config default}"

  python "${PROJECT_ROOT}/changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py" \
    --config "${CONFIG}" \
    --data_dir "${LMDB_DIR}" \
    --data_format lmdb \
    --out_dir "${OUT_DIR}" \
    --hidden_channels "${HIDDEN_CHANNELS}" \
    --num_res_blocks "${NUM_RES_BLOCKS}" \
    --scale_factor "${SCALE_FACTOR}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max_steps "${MAX_STEPS}" \
    "${ema_args[@]}" \
    --precision "${PRECISION}" \
    "${residual_args[@]}" \
    "${resume_args[@]}"
}

case "${MODE}" in
  check)
    check_lmdb
    check_python_deps
    check_stage2_import
    echo "Stage 2 distill LMDB training preflight passed: ${LMDB_DIR}"
    ;;
  train)
    train_stage2_distill
    ;;
  *)
    echo "Usage: bash changing_resolution_distill/scripts/train/run_clean_480p720p_stage2_distill_lmdb_training.sh [check|train]" >&2
    exit 2
    ;;
esac
