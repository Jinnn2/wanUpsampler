#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  check|prepare|plan|generate|finalize|all) ;;
  *) echo "Usage: $0 [check|prepare|plan|generate|finalize|all]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
PROTOCOL="${PROTOCOL:-${PROJECT_ROOT}/UNIV_adaptor/configs/univ_low_budget_extension.json}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/prompts/univ_controller_pilot_500.txt}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${PROJECT_ROOT}/UNIV_adaptor/configs/wan21_t2v_univ_rgb_720p.example.json}"
BASE_DATASET_ROOT="${BASE_DATASET_ROOT:-${PROJECT_ROOT}/outputs/univ_prompt_budget_full_v3}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_low_budget_extension_primary_v1}"
MANIFEST="${OUT_ROOT}/extension_manifest.json"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/run_low_budget_extension.py"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SPLITS="${SPLITS:-train,validation}"
JOB_CHUNK_SIZE="${JOB_CHUNK_SIZE:-25}"
MAX_JOBS_PER_WORKER="${MAX_JOBS_PER_WORKER:-0}"
RESUME="${RESUME:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-camera shake, overexposed, static image, blurry details, subtitles, text, watermark, low quality, jpeg artifacts, distorted hands, distorted face, malformed body, duplicate limbs}"
export DTYPE="${DTYPE:-BF16}"
GENERATION_LOCK_DIR=""

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight comma-separated ids." >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ || -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU_IDS must contain eight unique non-negative integers: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu}"]=1
done
for split in "${SPLIT_ARRAY[@]}"; do
  case "${split}" in train|validation|test) ;; *) echo "Unknown split: ${split}" >&2; exit 2 ;; esac
done
[[ "${MAX_JOBS_PER_WORKER}" =~ ^[0-9]+$ ]] || {
  echo "MAX_JOBS_PER_WORKER must be a non-negative integer." >&2
  exit 2
}

require_inputs() {
  [[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
  for path in "${PROTOCOL}" "${PROMPTS_FILE}" "${TEMPLATE_CONFIG}" "${DRIVER}"; do
    [[ -f "${path}" ]] || { echo "Required file not found: ${path}" >&2; exit 1; }
  done
  [[ -f "${BASE_DATASET_ROOT}/generation_manifest.json" ]] || {
    echo "Base dataset manifest not found: ${BASE_DATASET_ROOT}/generation_manifest.json" >&2
    exit 1
  }
  for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
    [[ -d "${path}" ]] || { echo "Required directory not found: ${path}" >&2; exit 1; }
  done
}

validate_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required." >&2; exit 1; }
  mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  declare -A AVAILABLE_SET=()
  for gpu in "${AVAILABLE_GPUS[@]}"; do AVAILABLE_SET["${gpu//[[:space:]]/}"]=1; done
  for gpu in "${GPU_ARRAY[@]}"; do
    [[ -n "${AVAILABLE_SET[${gpu}]:-}" ]] || { echo "GPU ${gpu} is unavailable." >&2; exit 1; }
  done
}

prepare_manifest() {
  require_inputs
  "${WAN_PYTHON}" "${DRIVER}" prepare \
    --protocol "${PROTOCOL}" \
    --prompts "${PROMPTS_FILE}" \
    --template-config "${TEMPLATE_CONFIG}" \
    --model-root "${MODEL_ROOT}" \
    --base-dataset-root "${BASE_DATASET_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --job-chunk-size "${JOB_CHUNK_SIZE}" \
    --worker-count 8
}

list_worker_jobs() {
  local slot="$1"
  local -a args=(
    "${DRIVER}" list-jobs
    --manifest "${MANIFEST}"
    --splits "${SPLIT_ARRAY[@]}"
    --worker-slot "${slot}"
  )
  (( MAX_JOBS_PER_WORKER > 0 )) && args+=(--limit "${MAX_JOBS_PER_WORKER}")
  "${WAN_PYTHON}" "${args[@]}"
}

print_plan() {
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  echo "UNIV low-budget extension plan"
  echo "  base:   ${BASE_DATASET_ROOT}"
  echo "  output: ${OUT_ROOT}"
  echo "  splits: ${SPLITS}"
  for slot in 0 1 2 3 4 5 6 7; do
    local count
    count="$(list_worker_jobs "${slot}" | wc -l)"
    echo "  GPU ${GPU_ARRAY[${slot}]}: ${count} jobs"
  done
}

generate_parallel() {
  require_inputs
  validate_gpus
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  mkdir -p "${OUT_ROOT}/logs/8gpu_data"
  GENERATION_LOCK_DIR="${OUT_ROOT}/.low_budget_generation.lock"
  if ! mkdir "${GENERATION_LOCK_DIR}" 2>/dev/null; then
    echo "Generation lock exists: ${GENERATION_LOCK_DIR}" >&2
    exit 1
  fi
  local -a pids=()
  cleanup() {
    if [[ -n "${GENERATION_LOCK_DIR:-}" ]]; then
      rmdir "${GENERATION_LOCK_DIR}" 2>/dev/null || true
    fi
  }
  stop_children() {
    trap - INT TERM
    for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    cleanup
    exit 130
  }
  trap stop_children INT TERM
  trap cleanup EXIT
  for slot in 0 1 2 3 4 5 6 7; do
    local gpu="${GPU_ARRAY[${slot}]}"
    local log="${OUT_ROOT}/logs/8gpu_data/gpu_${gpu}.log"
    (
      while IFS= read -r job_id; do
        [[ -n "${job_id}" ]] || continue
        args=(
          "${DRIVER}" generate-job
          --manifest "${MANIFEST}"
          --job-id "${job_id}"
          --wan-python "${WAN_PYTHON}"
          --lightx2v-repo "${LIGHTX2V_REPO}"
          --negative-prompt "${NEGATIVE_PROMPT}"
        )
        [[ "${RESUME}" == "1" ]] && args+=(--resume)
        CUDA_VISIBLE_DEVICES="${gpu}" \
          PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
          "${WAN_PYTHON}" "${args[@]}"
      done < <(list_worker_jobs "${slot}")
    ) >>"${log}" 2>&1 &
    pids+=("$!")
    echo "[launch] GPU ${gpu}, worker ${slot} -> ${log}"
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  cleanup
  GENERATION_LOCK_DIR=""
  trap - EXIT INT TERM
  (( failed == 0 )) || { echo "At least one worker failed; inspect logs." >&2; exit 1; }
}

finalize_records() {
  "${WAN_PYTHON}" "${DRIVER}" finalize \
    --manifest "${MANIFEST}" \
    --out-root "${OUT_ROOT}" \
    --splits "${SPLIT_ARRAY[@]}"
}

case "${MODE}" in
  check)
    require_inputs
    validate_gpus
    PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${WAN_PYTHON}" - "${MODEL_ROOT}" "${PROTOCOL}" <<'PY'
import json
import sys
import torch

from UNIV_adaptor.low_budget_protocol import validate_protocol
from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root
from lightx2v.common.ops import *  # noqa: F403
import UNIV_adaptor.mrflow_ablation_runner  # noqa: F401
from lightx2v.utils.registry_factory import RUNNER_REGISTER

validate_wan21_t2v_model_root(sys.argv[1])
validate_protocol(json.load(open(sys.argv[2], encoding="utf-8")))
if "wan2.1_univ_mrflow_budget" not in RUNNER_REGISTER:
    raise SystemExit("wan2.1_univ_mrflow_budget registration missing")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("Low-budget protocol, runtime imports, and CUDA passed")
PY
    ;;
  prepare) prepare_manifest ;;
  plan) prepare_manifest; print_plan ;;
  generate) print_plan; generate_parallel ;;
  finalize) finalize_records ;;
  all) prepare_manifest; print_plan; generate_parallel; finalize_records ;;
esac
