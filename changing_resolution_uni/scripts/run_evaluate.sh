#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_uni_clean_v1_1k_fresh_ema0999}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean_v1_1k}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_OUT_DIR}/last.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-step_*.pt}"
MODE="${MODE:-latent}"
OUT_DIR="${OUT_DIR:-${TRAIN_OUT_DIR}/evaluation/${MODE}}"
GPU_IDS="${GPU_IDS:-0}"
PRECISION="${PRECISION:-bf16}"
METHODS="${METHODS:-}"
MAX_SOURCES="${MAX_SOURCES:-0}"
SOURCE_OFFSET="${SOURCE_OFFSET:-0}"
SPLIT="${SPLIT:-val}"
MANIFEST="${MANIFEST:-}"
DECODE_MAX_SOURCES="${DECODE_MAX_SOURCES:-20}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-1234}"
RESUME="${RESUME:-1}"
SAVE_VISUALS="${SAVE_VISUALS:-0}"
VISUAL_MAX_SOURCES="${VISUAL_MAX_SOURCES:-8}"
RGB_METRICS="${RGB_METRICS:-psnr ssim lpips}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-4}"
FPS="${FPS:-16}"
PANEL_WIDTH="${PANEL_WIDTH:-416}"
PANEL_HEIGHT="${PANEL_HEIGHT:-240}"
TIMING_WARMUP="${TIMING_WARMUP:-5}"
TIMING_REPEATS="${TIMING_REPEATS:-20}"
INCLUDE_LAST="${INCLUDE_LAST:-0}"
RUN_LOG="${RUN_LOG:-${OUT_DIR}/evaluate_${MODE}.log}"

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
VAE_BACKEND="${VAE_BACKEND:-lightx2v}"

SPECIALIST_CHECKPOINT="${SPECIALIST_CHECKPOINT:-}"
SPECIALIST_CONFIG="${SPECIALIST_CONFIG:-}"
SPECIALIST_USE_EMA="${SPECIALIST_USE_EMA:-0}"

if [[ -z "${METHODS}" ]]; then
  if [[ "${MODE}" == "sweep" ]]; then
    METHODS="raw ema"
  else
    METHODS="raw ema nearest trilinear bicubic"
  fi
fi
if [[ "${MODE}" == "sweep" && -z "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="${TRAIN_OUT_DIR}"
fi

[[ -d "${DATA_DIR}" ]] || { echo "Evaluation LMDB not found: ${DATA_DIR}" >&2; exit 2; }
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  [[ -d "${CHECKPOINT_DIR}" ]] || { echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2; exit 2; }
else
  [[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 2; }
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
read -r -a METHOD_ARRAY <<< "${METHODS}"
read -r -a RGB_METRIC_ARRAY <<< "${RGB_METRICS}"
NUM_GPUS="${#GPUS[@]}"
(( NUM_GPUS > 0 )) || { echo "GPU_IDS must contain at least one GPU" >&2; exit 2; }

mkdir -p "${OUT_DIR}" "$(dirname "${RUN_LOG}")"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

base_args=(
  -m changing_resolution_uni.evaluate
  --mode "${MODE}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --split "${SPLIT}"
  --precision "${PRECISION}"
  --methods "${METHOD_ARRAY[@]}"
  --source_offset "${SOURCE_OFFSET}"
  --max_sources "${MAX_SOURCES}"
  --decode_max_sources "${DECODE_MAX_SOURCES}"
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
  --bootstrap_seed "${BOOTSTRAP_SEED}"
  --visual_max_sources "${VISUAL_MAX_SOURCES}"
  --rgb_metrics "${RGB_METRIC_ARRAY[@]}"
  --metric_batch_size "${METRIC_BATCH_SIZE}"
  --fps "${FPS}"
  --panel_width "${PANEL_WIDTH}"
  --panel_height "${PANEL_HEIGHT}"
  --timing_warmup "${TIMING_WARMUP}"
  --timing_repeats "${TIMING_REPEATS}"
)
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  base_args+=(--checkpoint_dir "${CHECKPOINT_DIR}" --checkpoint_glob "${CHECKPOINT_GLOB}")
  if [[ "${INCLUDE_LAST}" == "1" ]]; then base_args+=(--include_last); fi
else
  base_args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${MANIFEST}" ]]; then base_args+=(--manifest "${MANIFEST}"); fi
if [[ "${RESUME}" == "1" ]]; then base_args+=(--resume); fi
if [[ "${SAVE_VISUALS}" == "1" ]]; then base_args+=(--save_visuals); fi

if [[ "${MODE}" == "rgb" || "${MODE}" == "all" ]]; then
  [[ -d "${MODEL_ROOT}" ]] || { echo "Wan model root not found: ${MODEL_ROOT}" >&2; exit 2; }
  [[ -f "${VAE_PATH}" ]] || { echo "Wan VAE checkpoint not found: ${VAE_PATH}" >&2; exit 2; }
  base_args+=(
    --model_root "${MODEL_ROOT}"
    --vae_path "${VAE_PATH}"
    --wan_repo "${LIGHTX2V_REPO}"
    --vae_backend "${VAE_BACKEND}"
  )
fi
if [[ " ${METHODS} " == *" specialist "* ]]; then
  [[ -f "${SPECIALIST_CHECKPOINT}" ]] || { echo "Specialist checkpoint not found: ${SPECIALIST_CHECKPOINT}" >&2; exit 2; }
  base_args+=(--specialist_checkpoint "${SPECIALIST_CHECKPOINT}")
  if [[ -n "${SPECIALIST_CONFIG}" ]]; then base_args+=(--specialist_config "${SPECIALIST_CONFIG}"); fi
  if [[ "${SPECIALIST_USE_EMA}" == "1" ]]; then base_args+=(--specialist_use_ema); fi
fi

launcher=(python)
if (( NUM_GPUS > 1 )); then
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
fi

echo "U-ITU evaluation"
echo "  mode        : ${MODE}"
echo "  data        : ${DATA_DIR}"
echo "  checkpoint  : ${CHECKPOINT_DIR:-${CHECKPOINT}}"
echo "  output      : ${OUT_DIR}"
echo "  methods     : ${METHODS}"
echo "  GPUs        : ${GPU_IDS} (${NUM_GPUS})"
echo "  precision   : ${PRECISION}"
echo "  log         : ${RUN_LOG}"

cd "${PROJECT_ROOT}"
"${launcher[@]}" "${base_args[@]}" 2>&1 | tee -a "${RUN_LOG}"

# Quality evaluation can use every GPU, but latency must run on one isolated
# process. The Python evaluator deliberately skips timing under torchrun.
if [[ "${MODE}" == "all" && "${NUM_GPUS}" -gt 1 ]]; then
  timing_args=("${base_args[@]}")
  timing_args[3]="timing"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "${timing_args[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

echo "U-ITU evaluation ready: ${OUT_DIR}"
