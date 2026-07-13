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
CONFIG="${CR_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml}"
OUT_DIR="${CR_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p_stage2_lmdb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be >= 1, got ${NUM_GPUS}." >&2
  exit 2
fi
if (( NUM_GPUS > ${#VISIBLE_GPUS[@]} )); then
  echo "NUM_GPUS=${NUM_GPUS} exceeds CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi

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
  if (( NUM_GPUS > 1 )) && ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun not found. Install PyTorch torchrun or set NUM_GPUS=1." >&2
    exit 1
  fi
}

check_stage2_import() {
  python - <<'PY'
import torch
from wan_sr.models import WanCleanLatentResizerStage2
model = WanCleanLatentResizerStage2(hidden_channels=32, num_res_blocks=2)
x = torch.randn(1, 16, 2, 4, 6)
y = model(x, output_size=(6, 9))
if tuple(y.shape) != (1, 16, 2, 6, 9):
    raise SystemExit(f"Unexpected Stage 2 smoke-test output shape: {tuple(y.shape)}")
print(f"Stage 2 model smoke test passed: shape={tuple(y.shape)} params={sum(p.numel() for p in model.parameters())}")
PY
}

train_stage2() {
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

  echo "Stage 2 clean latent LMDB training"
  echo "  project : ${PROJECT_ROOT}"
  echo "  lmdb    : ${LMDB_DIR}"
  echo "  config  : ${CONFIG}"
  echo "  out_dir : ${OUT_DIR}"
  echo "  gpus    : ${CUDA_VISIBLE_DEVICES} (world_size=${NUM_GPUS})"
  echo "  steps   : ${MAX_STEPS}"
  echo "  effective batch: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"

  local launcher=(python)
  if (( NUM_GPUS > 1 )); then
    launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
  fi

  "${launcher[@]}" "${PROJECT_ROOT}/changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py" \
    --config "${CONFIG}" \
    --data_dir "${LMDB_DIR}" \
    --data_format lmdb \
    --out_dir "${OUT_DIR}" \
    --hidden_channels "${HIDDEN_CHANNELS}" \
    --num_res_blocks "${NUM_RES_BLOCKS}" \
    --scale_factor "${SCALE_FACTOR}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --num_workers "${NUM_WORKERS}" \
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
    check_stage2_import
    echo "Stage 2 LMDB training preflight passed: ${LMDB_DIR}"
    ;;
  train)
    train_stage2
    ;;
  *)
    echo "Usage: bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh [check|train]" >&2
    exit 2
    ;;
esac
