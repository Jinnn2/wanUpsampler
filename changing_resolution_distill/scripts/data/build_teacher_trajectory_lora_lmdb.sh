#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
USER_MODEL_ROOT="${MODEL_ROOT:-}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
CR_DISTILL_MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CR_DISTILL_DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${CR_DISTILL_MODEL_ROOT}/distill_model.pt}"
CR_DISTILL_MODEL_ID="${CR_DISTILL_MODEL_ID:-lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
MODEL_ROOT="${USER_MODEL_ROOT:-${CR_DISTILL_MODEL_ROOT}}"
PROMPTS_FILE="${PROMPTS_FILE:-${CR_DISTILL_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}}"
OUT_DIR="${CR_DISTILL_TEACHER_TRAJ_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_teacher_trajectory_lora_14b_cfgdistill_5k_step3}"
CONFIG_JSON="${CR_DISTILL_STAGE3_X0PRED_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_480p.json}"

DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500 250}"
TRAIN_STEP_INDEX="${TRAIN_STEP_INDEX:-2}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
BASE_SEED="${BASE_SEED:-9500}"
NUM_FRAMES="${NUM_FRAMES:-81}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
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

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}" "${CR_DISTILL_DIT_CKPT}" "${PROMPTS_FILE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Path not found: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "Config not found: ${CONFIG_JSON}" >&2
  exit 1
fi

RUNTIME_CONFIG="${OUT_DIR%/}.runtime.json"
python - "${CONFIG_JSON}" "${RUNTIME_CONFIG}" "${CR_DISTILL_DIT_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, ckpt = map(Path, sys.argv[1:])
data = json.loads(src.read_text(encoding="utf-8"))
data["dit_original_ckpt"] = str(ckpt)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

args=(
  --prompts_file "${PROMPTS_FILE}"
  --out_dir "${OUT_DIR}"
  --distill_model_id "${CR_DISTILL_MODEL_ID}"
  --lightx2v_repo "${LIGHTX2V_REPO}"
  --model_path "${MODEL_ROOT}"
  --config_json "${RUNTIME_CONFIG}"
  --model_cls "wan2.1_distill"
  --task "t2v"
  --denoising_step_list ${DENOISING_STEP_LIST}
  --train_step_index "${TRAIN_STEP_INDEX}"
  --sample_shift "${SAMPLE_SHIFT}"
  --sample_guide_scale "${GUIDE_SCALE}"
  --num_frames "${NUM_FRAMES}"
  --prompt_offset "${PROMPT_OFFSET}"
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

echo "Teacher-only LoRA LMDB build"
echo "  project          : ${PROJECT_ROOT}"
echo "  prompts          : ${PROMPTS_FILE}"
echo "  out_dir          : ${OUT_DIR}"
echo "  train_step_index : ${TRAIN_STEP_INDEX} (0-based; 2 captures x before step3)"
echo "  denoise list     : ${DENOISING_STEP_LIST}"
echo "  distill id       : ${CR_DISTILL_MODEL_ID}"
echo "  model            : ${MODEL_ROOT}"
echo "  dit ckpt         : ${CR_DISTILL_DIT_CKPT}"
echo "  config           : ${RUNTIME_CONFIG}"
echo "  gpu              : ${CUDA_VISIBLE_DEVICES}"

python "${PROJECT_ROOT}/changing_resolution_distill/scripts/data/build_teacher_trajectory_lora_lmdb.py" "${args[@]}"
python "${PROJECT_ROOT}/changing_resolution_distill/scripts/data/check_teacher_trajectory_lora_lmdb.py" "${OUT_DIR}" --samples 3
