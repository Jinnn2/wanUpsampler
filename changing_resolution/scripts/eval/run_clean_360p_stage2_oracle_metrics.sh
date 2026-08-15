#!/usr/bin/env bash
set -euo pipefail

# Evaluate the TAA-free oracle candidates and compile sample-level timestep labels.
# Formal input: 10 prompts x 13 candidates (30, 35, 40..50) plus Native-HR.
#
# Modes:
#   check    validate protocol, manifests, NFE, TAA state, and all videos
#   prepare  additionally write the per-case VBench custom-input prompt maps
#   run      run VBench (CASES may select a resumable subset)
#   collect  compile complete VBench outputs and timing manifests into labels
#   all      prepare, run every case, and collect

MODE="${1:-all}"
case "${MODE}" in
  check|prepare|run|collect|all) ;;
  *) echo "Usage: $0 [check|prepare|run|collect|all]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

ORACLE_ROOT="${ORACLE_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_taa_free_oracle_branch_360p}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
NGPUS="${NGPUS:-1}"
INCLUDE_OVERALL="${INCLUDE_OVERALL:-0}"
OVERALL_WEIGHT="${OVERALL_WEIGHT:-0}"
MAX_QUALITY_DROP="${MAX_QUALITY_DROP:-0.02}"
LATENCY_LAMBDA="${LATENCY_LAMBDA:-0.05}"
TIMING_SOURCE="${TIMING_SOURCE:-branch}"
STRICT_PROTOCOL="${STRICT_PROTOCOL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MIN_VIDEO_BYTES="${MIN_VIDEO_BYTES:-1024}"
CASES="${CASES:-}"

for name in INCLUDE_OVERALL STRICT_PROTOCOL SKIP_EXISTING; do
  value="${!name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1." >&2
    exit 2
  fi
done
if [[ ! "${NGPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NGPUS must be a positive integer." >&2
  exit 2
fi
case "${TIMING_SOURCE}" in
  branch|independent|prefer-independent) ;;
  *) echo "TIMING_SOURCE must be branch, independent, or prefer-independent." >&2; exit 2 ;;
esac
if [[ -n "${CASES}" && "${MODE}" != "run" ]]; then
  echo "CASES is supported only in run mode; collect/all require all 14 cases." >&2
  exit 2
fi
if [[ ! -x "${VBENCH_PYTHON}" ]]; then
  echo "VBENCH_PYTHON is not executable: ${VBENCH_PYTHON}" >&2
  exit 1
fi
if [[ "${MODE}" == "run" || "${MODE}" == "all" ]]; then
  [[ -f "${VBENCH_ROOT}/evaluate.py" ]] || {
    echo "Official VBench evaluate.py not found: ${VBENCH_ROOT}/evaluate.py" >&2
    exit 1
  }
fi

args=(
  "${MODE}"
  --oracle-root "${ORACLE_ROOT}"
  --vbench-root "${VBENCH_ROOT}"
  --python "${VBENCH_PYTHON}"
  --ngpus "${NGPUS}"
  --overall-weight "${OVERALL_WEIGHT}"
  --max-quality-drop "${MAX_QUALITY_DROP}"
  --latency-lambda "${LATENCY_LAMBDA}"
  --timing-source "${TIMING_SOURCE}"
  --min-video-bytes "${MIN_VIDEO_BYTES}"
)
if [[ "${INCLUDE_OVERALL}" == "1" ]]; then
  args+=(--include-overall)
fi
if [[ "${STRICT_PROTOCOL}" == "1" ]]; then
  args+=(--strict-protocol)
else
  args+=(--no-strict-protocol)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip-existing)
fi
if [[ -n "${CASES}" ]]; then
  read -r -a selected_cases <<< "${CASES}"
  args+=(--cases "${selected_cases[@]}")
fi

echo "[oracle-metrics] mode=${MODE} root=${ORACLE_ROOT}"
echo "[oracle-metrics] VBench-5=subject/background/motion/aesthetic/imaging; Dynamic Degree excluded"
echo "[oracle-metrics] overall=${INCLUDE_OVERALL} weight=${OVERALL_WEIGHT} timing=${TIMING_SOURCE} ngpus=${NGPUS}"

"${VBENCH_PYTHON}" \
  "${PROJECT_ROOT}/changing_resolution/scripts/eval/run_clean_360p_stage2_oracle_metrics.py" \
  "${args[@]}"

if [[ "${MODE}" == "collect" || "${MODE}" == "all" ]]; then
  echo "[oracle-metrics] labels=${ORACLE_ROOT}/metrics/oracle_labels.csv"
  echo "[oracle-metrics] canonical=${ORACLE_ROOT}/metrics/oracle_metrics.json"
fi
