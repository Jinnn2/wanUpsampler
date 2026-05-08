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
LMDB_DIR="${CR_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CONFIG="${CR_STAGE1_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage1.yaml}"
OUT_DIR="${CR_STAGE1_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p_stage1_lmdb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAX_STEPS="${MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
PRECISION="${PRECISION:-bf16}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-8}"
SCALE_FACTOR="${SCALE_FACTOR:-1.5}"
RESUME="${RESUME:-}"

export CUDA_VISIBLE_DEVICES
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

MODE="${1:-train}"

check_lmdb() {
  if [[ ! -d "${LMDB_DIR}" ]] || [[ -z "$(find "${LMDB_DIR}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
    echo "No LMDB shards found under: ${LMDB_DIR}" >&2
    echo "If raw videos are already generated in part_00..part_03, build LMDB first:" >&2
    echo "  TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 bash changing_resolution/scripts/data/build_clean_480p720p_lmdb_1k_multigpu.sh lmdb" >&2
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

train_stage1() {
  check_lmdb
  check_python_deps
  mkdir -p "${OUT_DIR}"

  local resume_args=()
  if [[ -n "${RESUME}" ]]; then
    resume_args=(--resume "${RESUME}")
  fi

  echo "Stage 1 clean latent LMDB baseline training"
  echo "  project : ${PROJECT_ROOT}"
  echo "  lmdb    : ${LMDB_DIR}"
  echo "  config  : ${CONFIG}"
  echo "  out_dir : ${OUT_DIR}"
  echo "  gpu     : ${CUDA_VISIBLE_DEVICES}"
  echo "  steps   : ${MAX_STEPS}"

  python "${PROJECT_ROOT}/changing_resolution/scripts/train/train_clean_latent_resizer.py" \
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
    --precision "${PRECISION}" \
    "${resume_args[@]}"
}

case "${MODE}" in
  check)
    check_lmdb
    check_python_deps
    echo "Stage 1 LMDB training preflight passed: ${LMDB_DIR}"
    ;;
  train)
    train_stage1
    ;;
  *)
    echo "Usage: bash changing_resolution/scripts/train/run_clean_480p720p_stage1_lmdb_training.sh [check|train]" >&2
    exit 2
    ;;
esac
