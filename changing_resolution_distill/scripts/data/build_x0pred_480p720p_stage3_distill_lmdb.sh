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
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
SOURCE_LMDB="${CR_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
HANDOFF_STEP="${HANDOFF_STEP:-2}"
OUT_DIR="${CR_DISTILL_STAGE3_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_distill_step${HANDOFF_STEP}}"
CONFIG_JSON="${CR_DISTILL_STAGE3_X0PRED_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_480p.json}"

MODE="${MODE:-lightx2v_distill}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500 250}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
BASE_SEED="${BASE_SEED:-9400}"
NUM_FRAMES="${NUM_FRAMES:-81}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
REQUIRE_SAMPLES="${REQUIRE_SAMPLES:-}"
SHARD_SIZE="${SHARD_SIZE:-100}"
MAP_SIZE_GB="${MAP_SIZE_GB:-256}"
PRECISION="${PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  fp32) export DTYPE="${DTYPE:-FP32}" ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16, fp16, or fp32" >&2
    exit 2
    ;;
esac

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}" "${SOURCE_LMDB}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Path not found: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "Config not found: ${CONFIG_JSON}" >&2
  exit 1
fi

args=(
  --source_lmdb "${SOURCE_LMDB}"
  --out_dir "${OUT_DIR}"
  --mode "${MODE}"
  --lightx2v_repo "${LIGHTX2V_REPO}"
  --model_path "${MODEL_ROOT}"
  --config_json "${CONFIG_JSON}"
  --model_cls "wan2.1_distill"
  --task "t2v"
  --denoising_step_list ${DENOISING_STEP_LIST}
  --handoff_step "${HANDOFF_STEP}"
  --sample_shift "${SAMPLE_SHIFT}"
  --sample_guide_scale "${GUIDE_SCALE}"
  --num_frames "${NUM_FRAMES}"
  --offset "${SAMPLE_OFFSET}"
  --shard_size "${SHARD_SIZE}"
  --map_size_gb "${MAP_SIZE_GB}"
  --base_seed "${BASE_SEED}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  args+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${REQUIRE_SAMPLES}" ]]; then
  args+=(--require_samples "${REQUIRE_SAMPLES}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${ENABLE_CFG:-0}" == "1" ]]; then
  args+=(--enable_cfg)
fi

echo "Stage 3 distill x0-pred LMDB build"
echo "  project      : ${PROJECT_ROOT}"
echo "  source_lmdb  : ${SOURCE_LMDB}"
echo "  out_dir      : ${OUT_DIR}"
echo "  handoff_step : ${HANDOFF_STEP}"
echo "  denoise list : ${DENOISING_STEP_LIST}"
echo "  model        : ${MODEL_ROOT}"
echo "  config       : ${CONFIG_JSON}"
echo "  gpu          : ${CUDA_VISIBLE_DEVICES}"

python "${PROJECT_ROOT}/changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb.py" "${args[@]}"
