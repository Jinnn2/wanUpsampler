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
MODEL_ROOT="${USER_MODEL_ROOT:-${CR_DISTILL_MODEL_ROOT}}"
GEN_CONFIG="${CR_DISTILL_GENERATE_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_generate_720p.json}"
PROMPTS_FILE="${CR_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}"
OUT_DIR="${CR_DISTILL_RAW_VIDEO_DIR_1K:-${PROJECT_ROOT}/data/changing_resolution_distill/raw_wan21_14b_cfgdistill_720p_1k}"
START_SEED="${START_SEED:-620000}"
MAX_PROMPTS="${MAX_PROMPTS:-1000}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

export CUDA_VISIBLE_DEVICES

if [[ ! -d "${LIGHTX2V_REPO}" ]]; then
  echo "LightX2V repo not found: ${LIGHTX2V_REPO}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "Wan distill model root not found: ${MODEL_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${GEN_CONFIG}" ]]; then
  echo "Generation config not found: ${GEN_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Prompts file not found: ${PROMPTS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

lightx2v_path="${LIGHTX2V_REPO}"
model_path="${MODEL_ROOT}"
# shellcheck source=/dev/null
source "${LIGHTX2V_REPO}/scripts/base/base.sh"
export PYTHONPATH="${PROJECT_ROOT}:${LIGHTX2V_REPO}:${PYTHONPATH:-}"

index=0
prompt_index=0
while IFS= read -r prompt || [[ -n "${prompt}" ]]; do
  [[ -z "${prompt}" || "${prompt}" == \#* ]] && continue
  if (( prompt_index < PROMPT_OFFSET )); then
    prompt_index=$((prompt_index + 1))
    continue
  fi
  if (( index >= MAX_PROMPTS )); then
    break
  fi

  seed=$((START_SEED + index))
  sample_id="$(printf "%06d" "$((PROMPT_OFFSET + index))")"
  out_path="${OUT_DIR}/wan21_14b_cfgdistill_720p_${sample_id}_seed${seed}.mp4"

  if [[ -f "${out_path}" ]] && ffprobe -v error "${out_path}" >/dev/null 2>&1; then
    echo "Skip existing valid video: ${out_path}"
  else
    echo "Generating ${out_path}"
    python -m lightx2v.infer \
      --seed "${seed}" \
      --model_cls wan2.1_distill \
      --task t2v \
      --model_path "${MODEL_ROOT}" \
      --config_json "${GEN_CONFIG}" \
      --prompt "${prompt}" \
      --negative_prompt "${NEGATIVE_PROMPT}" \
      --save_result_path "${out_path}"
  fi

  index=$((index + 1))
  prompt_index=$((prompt_index + 1))
done < "${PROMPTS_FILE}"

count="$(find "${OUT_DIR}" -type f -name '*.mp4' | wc -l)"
echo "Wan2.1 14B CfgDistill generated 720p dataset ready: ${OUT_DIR} (${count} mp4 files)"
