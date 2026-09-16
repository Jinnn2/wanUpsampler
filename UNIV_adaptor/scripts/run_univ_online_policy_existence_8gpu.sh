#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  check|prepare|plan|generate|vbench|analyze|all) ;;
  *)
    echo "Usage: $0 [check|prepare|plan|generate|vbench|analyze|all]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
SPEC="${SPEC:-${PROJECT_ROOT}/UNIV_adaptor/configs/univ_online_policy_existence.json}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${PROJECT_ROOT}/UNIV_adaptor/configs/univ_mrflow_refinement_ablation.json}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/prompts/univ_controller_pilot_500.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_online_policy_existence_v1}"
MANIFEST="${OUT_ROOT}/generation_manifest.json"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/validation/run_online_policy_existence.py"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
VBENCH_GPU_IDS="${VBENCH_GPU_IDS:-${GPU_IDS}}"
VBENCH_NGPUS="${VBENCH_NGPUS:-8}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
LIMIT="${LIMIT:-16}"
TIMING_WARMUP="${TIMING_WARMUP:-1}"
SEED="${SEED:-9700}"
RESUME="${RESUME:-1}"
FORCE_VBENCH="${FORCE_VBENCH:-0}"
SKIP_VBENCH_WARMUP="${SKIP_VBENCH_WARMUP:-0}"
VBENCH_COMMIT="${VBENCH_COMMIT:-}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-camera shake, overexposed, static image, blurry details, subtitles, text, watermark, low quality, jpeg artifacts, distorted hands, distorted face, malformed body, duplicate limbs}"
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
for value in "${PROMPT_OFFSET}" "${TIMING_WARMUP}" "${BOOTSTRAP_REPETITIONS}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Offsets, warmup and bootstrap count must be integers." >&2; exit 2; }
done
[[ "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive." >&2; exit 2; }

require_generation_inputs() {
  [[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
  for path in "${SPEC}" "${TEMPLATE_CONFIG}" "${PROMPTS_FILE}" "${DRIVER}"; do
    [[ -f "${path}" ]] || { echo "Required file not found: ${path}" >&2; exit 1; }
  done
  for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
    [[ -d "${path}" ]] || { echo "Required directory not found: ${path}" >&2; exit 1; }
  done
}

validate_visible_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required." >&2; exit 1; }
  mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  declare -A AVAILABLE_SET=()
  for gpu in "${AVAILABLE_GPUS[@]}"; do AVAILABLE_SET["${gpu//[[:space:]]/}"]=1; done
  for gpu in "${GPU_ARRAY[@]}"; do
    [[ -n "${AVAILABLE_SET[${gpu}]:-}" ]] || { echo "GPU ${gpu} is unavailable." >&2; exit 1; }
  done
}

resolve_vbench_python() {
  [[ -f "${VBENCH_ROOT}/evaluate.py" ]] || { echo "Missing VBench checkout: ${VBENCH_ROOT}" >&2; exit 1; }
  if [[ -n "${VBENCH_PYTHON}" ]]; then
    [[ -x "${VBENCH_PYTHON}" ]] || { echo "VBENCH_PYTHON is not executable." >&2; exit 1; }
    return
  fi
  local candidate
  for candidate in /opt/conda/envs/vbench/bin/python /opt/conda/bin/python "$(command -v python 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if (cd "${VBENCH_ROOT}"; "${candidate}" -c 'import torch, vbench' >/dev/null 2>&1); then
      VBENCH_PYTHON="${candidate}"
      return
    fi
  done
  echo "Set VBENCH_PYTHON to an environment that imports torch and vbench." >&2
  exit 1
}

prepare_manifest() {
  require_generation_inputs
  "${WAN_PYTHON}" "${DRIVER}" prepare \
    --spec "${SPEC}" \
    --template-config "${TEMPLATE_CONFIG}" \
    --prompts "${PROMPTS_FILE}" \
    --out-root "${OUT_ROOT}" \
    --model-root "${MODEL_ROOT}" \
    --lightx2v-repo "${LIGHTX2V_REPO}" \
    --prompt-offset "${PROMPT_OFFSET}" \
    --limit "${LIMIT}" \
    --timing-warmup "${TIMING_WARMUP}" \
    --seed "${SEED}"
}

load_cases() {
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  mapfile -t CASE_ARRAY < <("${WAN_PYTHON}" "${DRIVER}" list-cases --manifest "${MANIFEST}")
  (( ${#CASE_ARRAY[@]} == 8 )) || { echo "Manifest must resolve to eight cases." >&2; exit 1; }
}

print_plan() {
  load_cases
  echo "UNIV online-policy existence pilot"
  echo "  prompts : ${LIMIT} at offset ${PROMPT_OFFSET}"
  echo "  output  : ${OUT_ROOT}"
  for slot in 0 1 2 3 4 5 6 7; do
    echo "  GPU ${GPU_ARRAY[${slot}]} -> ${CASE_ARRAY[${slot}]}"
  done
}

generate_parallel() {
  validate_visible_gpus
  load_cases
  mkdir -p "${OUT_ROOT}/logs/8gpu_generation"
  local lock_dir="${OUT_ROOT}/.online_policy_generation.lock"
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "Generation lock exists: ${lock_dir}" >&2
    exit 1
  fi
  local -a pids=()
  local -a labels=()
  cleanup_lock() { rmdir "${lock_dir}" 2>/dev/null || true; }
  stop_children() {
    trap - INT TERM
    for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    cleanup_lock
    exit 130
  }
  trap stop_children INT TERM
  trap cleanup_lock EXIT
  for slot in 0 1 2 3 4 5 6 7; do
    local gpu="${GPU_ARRAY[${slot}]}"
    local case_name="${CASE_ARRAY[${slot}]}"
    local log="${OUT_ROOT}/logs/8gpu_generation/${case_name}.log"
    (
      args=(
        "${DRIVER}" generate-case
        --manifest "${MANIFEST}"
        --case-name "${case_name}"
        --wan-python "${WAN_PYTHON}"
        --lightx2v-repo "${LIGHTX2V_REPO}"
        --negative-prompt "${NEGATIVE_PROMPT}"
      )
      [[ "${RESUME}" == "1" ]] && args+=(--resume)
      CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${WAN_PYTHON}" "${args[@]}"
    ) >>"${log}" 2>&1 &
    pids+=("$!")
    labels+=("${case_name}")
    echo "[launch] GPU ${gpu}: ${case_name} -> ${log}"
  done
  local failed=0
  for slot in "${!pids[@]}"; do
    if wait "${pids[${slot}]}"; then
      echo "[complete] ${labels[${slot}]}"
    else
      echo "[failed] ${labels[${slot}]}" >&2
      failed=1
    fi
  done
  cleanup_lock
  trap - EXIT INT TERM
  (( failed == 0 )) || { echo "At least one lane failed; completed cases remain resumable." >&2; exit 1; }
}

score_vbench() {
  resolve_vbench_python
  args=(
    "${DRIVER}" vbench
    --manifest "${MANIFEST}"
    --vbench-root "${VBENCH_ROOT}"
    --vbench-python "${VBENCH_PYTHON}"
    --vbench-ngpus "${VBENCH_NGPUS}"
  )
  [[ -n "${VBENCH_COMMIT}" ]] && args+=(--vbench-commit "${VBENCH_COMMIT}")
  [[ "${SKIP_VBENCH_WARMUP}" == "1" ]] && args+=(--skip-vbench-warmup)
  [[ "${FORCE_VBENCH}" == "1" ]] && args+=(--force)
  CUDA_VISIBLE_DEVICES="${VBENCH_GPU_IDS}" \
    PYTHONPATH="${VBENCH_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${WAN_PYTHON}" "${args[@]}"
}

analyze_results() {
  "${WAN_PYTHON}" "${DRIVER}" analyze \
    --manifest "${MANIFEST}" \
    --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"
}

case "${MODE}" in
  check)
    require_generation_inputs
    validate_visible_gpus
    resolve_vbench_python
    PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${WAN_PYTHON}" - "${MODEL_ROOT}" "${SPEC}" <<'PY'
import json
import sys
import torch
from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root
from UNIV_adaptor.online_policy import validate_spec
from lightx2v.common.ops import *  # noqa: F403
import UNIV_adaptor.online_policy_runner  # noqa: F401
from lightx2v.utils.registry_factory import RUNNER_REGISTER

validate_wan21_t2v_model_root(sys.argv[1])
validate_spec(json.load(open(sys.argv[2], encoding="utf-8")))
if "wan2.1_univ_online_policy_existence" not in RUNNER_REGISTER:
    raise SystemExit("online policy runner registration missing")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("Online-policy spec, Wan model, runner imports and CUDA passed")
PY
    (cd "${VBENCH_ROOT}"; "${VBENCH_PYTHON}" -c 'import torch, vbench; assert torch.cuda.is_available()')
    ;;
  prepare) prepare_manifest ;;
  plan) prepare_manifest; print_plan ;;
  generate) prepare_manifest; print_plan; generate_parallel ;;
  vbench) score_vbench ;;
  analyze) analyze_results ;;
  all)
    prepare_manifest
    print_plan
    generate_parallel
    score_vbench
    analyze_results
    ;;
esac

echo "Manifest: ${MANIFEST}"
echo "Report  : ${OUT_ROOT}/reports/POLICY_EXISTENCE.md"
