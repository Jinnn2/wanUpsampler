#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
USER_TAIL_SKIP_LMDB_DIR="${TAIL_SKIP_LORA_LMDB_DIR+x}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/path/to/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/Wan-AI/Wan2.1-T2V-1.3B}"
SOURCE_LMDB="${CR_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CONFIG_JSON="${TAIL_SKIP_LORA_CONFIG_JSON:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_x0pred_480p.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INFER_STEPS="${INFER_STEPS:-50}"
TRAIN_STEP="${TRAIN_STEP:-45}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
ENABLE_CFG="${ENABLE_CFG:-1}"
if [[ -n "${USER_TAIL_SKIP_LMDB_DIR}" ]]; then
  OUT_DIR="${TAIL_SKIP_LORA_LMDB_DIR}"
else
  OUT_DIR="${PROJECT_ROOT}/data/changing_resolution/lmdb_tail_skip_lora_step${TRAIN_STEP}_to_step${INFER_STEPS}"
fi
BASE_SEED="${BASE_SEED:-9400}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
MODE="${MODE:-lightx2v}"
PRECISION="${PRECISION:-bf16}"
OVERWRITE="${OVERWRITE:-0}"

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

if [[ ! -d "${SOURCE_LMDB}" ]] || [[ -z "$(find "${SOURCE_LMDB}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
  echo "No source clean LMDB shards found under: ${SOURCE_LMDB}" >&2
  exit 1
fi
if [[ "${MODE}" == "lightx2v" ]]; then
  for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}" "${CONFIG_JSON}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Required path not found: ${path}" >&2
      exit 1
    fi
  done
fi

extra_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  extra_args+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ "${ENABLE_CFG}" == "1" ]]; then
  extra_args+=(--enable_cfg)
else
  extra_args+=(--no-enable_cfg)
fi

python -m py_compile "${PROJECT_ROOT}/changing_resolution/scripts/data/build_tail_skip_lora_lmdb.py"

echo "Tail-skip LoRA LMDB build"
echo "  source_lmdb : ${SOURCE_LMDB}"
echo "  out_dir     : ${OUT_DIR}"
echo "  train_step  : ${TRAIN_STEP}"
echo "  infer_steps : ${INFER_STEPS}"
echo "  mode        : ${MODE}"
echo "  enable_cfg  : ${ENABLE_CFG}"

python "${PROJECT_ROOT}/changing_resolution/scripts/data/build_tail_skip_lora_lmdb.py" \
  --source_lmdb "${SOURCE_LMDB}" \
  --out_dir "${OUT_DIR}" \
  --mode "${MODE}" \
  --lightx2v_repo "${LIGHTX2V_REPO}" \
  --model_path "${MODEL_ROOT}" \
  --config_json "${CONFIG_JSON}" \
  --infer_steps "${INFER_STEPS}" \
  --train_step "${TRAIN_STEP}" \
  --sample_shift "${SAMPLE_SHIFT}" \
  --sample_guide_scale "${GUIDE_SCALE}" \
  --base_seed "${BASE_SEED}" \
  --offset "${SAMPLE_OFFSET}" \
  --precision "${PRECISION}" \
  "${extra_args[@]}"
