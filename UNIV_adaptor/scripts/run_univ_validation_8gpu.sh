#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  check|plan|prepare|generate|postprocess|all) ;;
  *) echo "Usage: $0 [check|plan|prepare|generate|postprocess|all]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SINGLE_GPU_SUITE="${SCRIPT_DIR}/run_univ_validation_suite.sh"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PROFILE="${PROFILE:-core}"
LIMIT="${LIMIT:-10}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_validation_core_${LIMIT}p_8gpu}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs/8gpu_generation}"
MANIFEST="${OUT_ROOT}/run_manifest.json"

if [[ "${PROFILE}" != "core" ]]; then
  echo "The first 8-GPU launcher requires PROFILE=core (exactly eight cases)." >&2
  exit 2
fi
[[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
[[ -f "${SINGLE_GPU_SUITE}" ]] || {
  echo "Validation suite not found: ${SINGLE_GPU_SUITE}" >&2
  exit 1
}

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight comma-separated device ids: ${GPU_IDS}" >&2
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

validate_visible_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required for 8-GPU generation." >&2
    exit 1
  }
  mapfile -t AVAILABLE_GPUS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
  )
  declare -A AVAILABLE_SET=()
  for gpu in "${AVAILABLE_GPUS[@]}"; do
    AVAILABLE_SET["${gpu//[[:space:]]/}"]=1
  done
  for gpu in "${GPU_ARRAY[@]}"; do
    if [[ -z "${AVAILABLE_SET[${gpu}]:-}" ]]; then
      echo "Requested GPU ${gpu} is not reported by nvidia-smi." >&2
      exit 1
    fi
  done
}

run_single_suite() {
  local action="$1"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    WAN_PYTHON="${WAN_PYTHON}" \
    PROFILE="${PROFILE}" \
    LIMIT="${LIMIT}" \
    OUT_ROOT="${OUT_ROOT}" \
    CASE_NAME="" \
    bash "${SINGLE_GPU_SUITE}" "${action}"
}

prepare_manifest() {
  run_single_suite prepare
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not created: ${MANIFEST}" >&2; exit 1; }
}

load_cases() {
  [[ -f "${MANIFEST}" ]] || {
    echo "Manifest not found: ${MANIFEST}; run prepare first." >&2
    exit 1
  }
  mapfile -t CASE_ARRAY < <(
    "${WAN_PYTHON}" - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("profile") != "core":
    raise SystemExit("8-GPU launcher requires an immutable core manifest")
cases = manifest.get("cases", [])
if len(cases) != 8:
    raise SystemExit(f"core manifest must contain exactly 8 cases, got {len(cases)}")
for case in cases:
    print(case["name"])
PY
  )
  if (( ${#CASE_ARRAY[@]} != 8 )); then
    echo "Failed to resolve exactly eight cases from ${MANIFEST}." >&2
    exit 1
  fi
}

print_plan() {
  load_cases
  echo "UNIV 8-GPU fixed-action discovery plan"
  echo "  profile : ${PROFILE}"
  echo "  prompts : ${LIMIT}"
  echo "  output  : ${OUT_ROOT}"
  for index in "${!GPU_ARRAY[@]}"; do
    echo "  GPU ${GPU_ARRAY[${index}]} -> ${CASE_ARRAY[${index}]}"
  done
}

generate_parallel() {
  validate_visible_gpus
  load_cases
  mkdir -p "${LOG_DIR}"
  local lock_dir="${OUT_ROOT}/.univ_8gpu_generation.lock"
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "Generation lock already exists: ${lock_dir}" >&2
    echo "Remove it only after confirming that no prior 8-GPU launcher is running." >&2
    exit 1
  fi

  local -a pids=()
  local -a labels=()
  cleanup_lock() {
    rmdir "${lock_dir}" 2>/dev/null || true
  }
  stop_children() {
    trap - INT TERM
    for pid in "${pids[@]:-}"; do
      kill "${pid}" 2>/dev/null || true
    done
    cleanup_lock
    exit 130
  }
  trap stop_children INT TERM
  trap cleanup_lock EXIT

  for index in "${!GPU_ARRAY[@]}"; do
    local gpu="${GPU_ARRAY[${index}]}"
    local case_name="${CASE_ARRAY[${index}]}"
    local log_path="${LOG_DIR}/${case_name}.log"
    echo "[launch] GPU ${gpu}: ${case_name} -> ${log_path}"
    (
      env \
        PROJECT_ROOT="${PROJECT_ROOT}" \
        WAN_PYTHON="${WAN_PYTHON}" \
        PROFILE="${PROFILE}" \
        LIMIT="${LIMIT}" \
        OUT_ROOT="${OUT_ROOT}" \
        GPU_ID="${gpu}" \
        CASE_NAME="${case_name}" \
        bash "${SINGLE_GPU_SUITE}" generate
    ) >>"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${case_name}")
  done

  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
      echo "[complete] ${labels[${index}]}"
    else
      echo "[failed] ${labels[${index}]}: ${LOG_DIR}/${labels[${index}]}.log" >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "At least one GPU lane failed; completed lanes remain resumable." >&2
    exit 1
  fi
  echo "All eight generation lanes completed."
}

postprocess() {
  run_single_suite visualize
  run_single_suite vbench
  run_single_suite summarize
}

case "${MODE}" in
  check)
    validate_visible_gpus
    run_single_suite check
    ;;
  prepare)
    prepare_manifest
    ;;
  plan)
    prepare_manifest
    print_plan
    ;;
  generate)
    print_plan
    generate_parallel
    ;;
  postprocess)
    postprocess
    ;;
  all)
    prepare_manifest
    print_plan
    generate_parallel
    postprocess
    ;;
esac
