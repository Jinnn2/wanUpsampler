#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"
case "${MODE}" in
  check|prepare|reuse|plan|generate|finalize|score-check|score|embed|train|confirm|all) ;;
  *) echo "Usage: $0 [check|prepare|reuse|plan|generate|finalize|score-check|score|embed|train|confirm|all]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
REALESRGAN_REPO="${REALESRGAN_REPO:-/mnt/afs_2/houze/Real-ESRGAN}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
PROTOCOL="${PROTOCOL:-${PROJECT_ROOT}/UNIV_adaptor/configs/univ_controlled_factor_v1.json}"
PROMPTS_JSONL="${PROMPTS_JSONL:-${PROJECT_ROOT}/prompts/univ_controlled_factor_v1.jsonl}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${PROJECT_ROOT}/UNIV_adaptor/configs/wan21_t2v_univ_rgb_720p.example.json}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_controlled_factor_v1}"
REUSE_ROOT="${REUSE_ROOT:-${PROJECT_ROOT}/outputs/univ_targeted_st_temporal_calibration_v2}"
MANIFEST="${OUT_ROOT}/generation_manifest.json"
DATASET="${OUT_ROOT}/controlled_factor_dataset.json"
SCORED_DIR="${OUT_ROOT}/metrics/controlled_factor_vbench"
T5_DIR="${SCORED_DIR}/t5_controlled_factor"
TRAIN_OUT="${TRAIN_OUT:-${SCORED_DIR}/prompt_prior_${TRAIN_FEATURES:-t5}}"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/run_controlled_factor_generation.py"
SCORER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/score_controlled_factor_dataset.py"
TRAINER="${PROJECT_ROOT}/UNIV_adaptor/scripts/router/train_controlled_factor_prompt_prior.py"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
JOB_CHUNK_SIZE=8
MAX_JOBS_PER_WORKER="${MAX_JOBS_PER_WORKER:-0}"
RESUME="${RESUME:-1}"
IMPORT_REUSE="${IMPORT_REUSE:-1}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-}"
TRAIN_FEATURES="${TRAIN_FEATURES:-t5}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-camera shake, overexposed, blurry details, subtitles, readable text, watermark, low quality, jpeg artifacts, distorted hands, distorted face, malformed body, duplicate limbs}"
export DTYPE="${DTYPE:-BF16}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
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
[[ "${MAX_JOBS_PER_WORKER}" =~ ^[0-9]+$ ]] || { echo "MAX_JOBS_PER_WORKER must be non-negative." >&2; exit 2; }
[[ "${TRAIN_FEATURES}" == "tfidf" || "${TRAIN_FEATURES}" == "t5" ]] || { echo "TRAIN_FEATURES must be tfidf or t5." >&2; exit 2; }

require_generation_inputs() {
  [[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
  for path in "${PROTOCOL}" "${PROMPTS_JSONL}" "${TEMPLATE_CONFIG}" "${DRIVER}" "${SCORER}" "${TRAINER}"; do
    [[ -f "${path}" ]] || { echo "Required file not found: ${path}" >&2; exit 1; }
  done
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

check_protocol() {
  require_generation_inputs
  "${WAN_PYTHON}" "${DRIVER}" check \
    --protocol "${PROTOCOL}" \
    --prompts-jsonl "${PROMPTS_JSONL}" \
    --template-config "${TEMPLATE_CONFIG}" \
    --model-root "${MODEL_ROOT}"
}

prepare_plan() {
  require_generation_inputs
  "${WAN_PYTHON}" "${DRIVER}" prepare \
    --protocol "${PROTOCOL}" \
    --prompts-jsonl "${PROMPTS_JSONL}" \
    --template-config "${TEMPLATE_CONFIG}" \
    --model-root "${MODEL_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --job-chunk-size "${JOB_CHUNK_SIZE}"
}

import_reuse() {
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  if [[ "${IMPORT_REUSE}" == "1" && -f "${REUSE_ROOT}/targeted_st_dataset.json" ]]; then
    "${WAN_PYTHON}" "${DRIVER}" reuse-targeted --manifest "${MANIFEST}" --reuse-root "${REUSE_ROOT}"
  else
    echo "Targeted reuse skipped: IMPORT_REUSE=${IMPORT_REUSE}, root=${REUSE_ROOT}"
  fi
}

list_worker_jobs() {
  local slot="$1"
  local -a args=("${DRIVER}" list-jobs --manifest "${MANIFEST}" --worker-slot "${slot}")
  (( MAX_JOBS_PER_WORKER > 0 )) && args+=(--limit "${MAX_JOBS_PER_WORKER}")
  "${WAN_PYTHON}" "${args[@]}"
}

print_plan() {
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  echo "UNIV controlled prompt-factor plan"
  echo "  prompts: 80 (48 train / 16 validation / 16 locked test)"
  echo "  arms: FULL + spatial + temporal + cache"
  echo "  seeds: 3"
  echo "  total videos: 960 (up to 48 reused)"
  echo "  output: ${OUT_ROOT}"
  for slot in 0 1 2 3 4 5 6 7; do
    local count
    count="$(list_worker_jobs "${slot}" | wc -l)"
    echo "  GPU ${GPU_ARRAY[${slot}]}: ${count} jobs"
  done
}

generate_parallel() {
  require_generation_inputs
  validate_gpus
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  mkdir -p "${OUT_ROOT}/logs/8gpu_controlled_factor"
  local lock="${OUT_ROOT}/.controlled_factor_generation.lock"
  if ! mkdir "${lock}" 2>/dev/null; then echo "Generation lock exists: ${lock}" >&2; exit 1; fi
  local -a pids=()
  cleanup() { rmdir "${lock}" 2>/dev/null || true; }
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
    local log="${OUT_ROOT}/logs/8gpu_controlled_factor/gpu_${gpu}.log"
    (
      while IFS= read -r job_id; do
        [[ -n "${job_id}" ]] || continue
        args=(
          "${DRIVER}" generate-job --manifest "${MANIFEST}" --job-id "${job_id}"
          --wan-python "${WAN_PYTHON}" --lightx2v-repo "${LIGHTX2V_REPO}"
          --realesrgan-repo "${REALESRGAN_REPO}" --negative-prompt "${NEGATIVE_PROMPT}"
        )
        [[ "${RESUME}" == "1" ]] && args+=(--resume)
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
          "${WAN_PYTHON}" "${args[@]}"
      done < <(list_worker_jobs "${slot}")
    ) >>"${log}" 2>&1 &
    pids+=("$!")
    echo "[launch] GPU ${gpu}, worker ${slot} -> ${log}"
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  cleanup
  trap - EXIT INT TERM
  (( failed == 0 )) || { echo "At least one worker failed; inspect logs." >&2; exit 1; }
}

finalize_records() {
  "${WAN_PYTHON}" "${DRIVER}" finalize --manifest "${MANIFEST}" --out-root "${OUT_ROOT}"
}

score_dataset() {
  [[ -f "${DATASET}" ]] || { echo "Finalized dataset not found: ${DATASET}" >&2; exit 1; }
  [[ -x "${VBENCH_PYTHON}" ]] || { echo "VBench Python is not executable: ${VBENCH_PYTHON}" >&2; exit 1; }
  [[ -d "${VBENCH_ROOT}" ]] || { echo "VBench root not found: ${VBENCH_ROOT}" >&2; exit 1; }
  local commit="${EXPECTED_VBENCH_COMMIT}"
  [[ -n "${commit}" ]] || commit="$(git -C "${VBENCH_ROOT}" rev-parse HEAD)"
  local -a args=(
    "${SCORER}" all --dataset-root "${OUT_ROOT}" --vbench-root "${VBENCH_ROOT}"
    --vbench-python "${VBENCH_PYTHON}" --ngpus 8 --expected-vbench-commit "${commit}"
  )
  [[ "${FORCE_RESCORE}" == "1" ]] && args+=(--force-rescore)
  PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${VBENCH_PYTHON}" "${args[@]}"
}

trainer_args() {
  printf '%s\n' \
    "--scored-dir" "${SCORED_DIR}" \
    "--out-dir" "${TRAIN_OUT}" \
    "--features" "${TRAIN_FEATURES}" \
    "--t5-dir" "${T5_DIR}" \
    "--model-root" "${MODEL_ROOT}" \
    "--lightx2v-repo" "${LIGHTX2V_REPO}" \
    "--device" "${TRAIN_DEVICE}"
}

embed_prompts() {
  if [[ "${TRAIN_FEATURES}" == "tfidf" ]]; then
    echo "T5 embedding skipped because TRAIN_FEATURES=tfidf"
    return
  fi
  mapfile -t args < <(trainer_args)
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${WAN_PYTHON}" "${TRAINER}" embed "${args[@]}"
}

train_prior() {
  mapfile -t args < <(trainer_args)
  "${WAN_PYTHON}" "${TRAINER}" train "${args[@]}"
}

confirm_test() {
  [[ "${CONFIRM_TEST_ACCESS:-0}" == "1" ]] || {
    echo "Locked test confirmation requires CONFIRM_TEST_ACCESS=1." >&2
    exit 2
  }
  mapfile -t args < <(trainer_args)
  "${WAN_PYTHON}" "${TRAINER}" confirm "${args[@]}" --confirm-test-access
}

case "${MODE}" in
  check) check_protocol; validate_gpus ;;
  prepare) prepare_plan ;;
  reuse) import_reuse ;;
  plan) prepare_plan; import_reuse; print_plan ;;
  generate) print_plan; generate_parallel ;;
  finalize) finalize_records ;;
  score-check) "${VBENCH_PYTHON}" "${SCORER}" check --dataset-root "${OUT_ROOT}" ;;
  score) score_dataset ;;
  embed) embed_prompts ;;
  train) train_prior ;;
  confirm) confirm_test ;;
  all) prepare_plan; import_reuse; print_plan; generate_parallel; finalize_records; score_dataset; embed_prompts; train_prior ;;
esac
