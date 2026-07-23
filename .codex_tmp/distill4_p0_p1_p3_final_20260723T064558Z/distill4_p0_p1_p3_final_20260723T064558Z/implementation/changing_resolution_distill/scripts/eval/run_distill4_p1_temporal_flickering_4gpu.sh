#!/usr/bin/env bash
set -euo pipefail

# P1: evaluate temporal_flickering on an already generated Distill4 suite.
# Raw and canonical outputs are isolated from the original six-factor VBench run.

ACTION="${1:-run}"
if [[ "${ACTION}" != "prepare" && "${ACTION}" != "run" && "${ACTION}" != "collect" ]]; then
  echo "Usage: $0 [prepare|run|collect]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
FACTORIAL_ROOT="${DISTILL4_FINAL_QUALITY_EFFICIENCY:-${PROJECT_ROOT}/outputs/aaai27_experiments/quality_efficiency_distill4}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/envs/vbench/bin/python}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 4 )); then
  echo "Exactly four comma-separated GPU ids are required; got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi
[[ -x "${VBENCH_PYTHON}" ]] || { echo "Python is not executable: ${VBENCH_PYTHON}" >&2; exit 1; }
[[ -f "${FACTORIAL_ROOT}/run_manifest.json" ]] || { echo "Missing run manifest: ${FACTORIAL_ROOT}/run_manifest.json" >&2; exit 1; }
if [[ "${ACTION}" == "run" ]]; then
  [[ -f "${VBENCH_ROOT}/evaluate.py" ]] || { echo "Official VBench evaluate.py not found: ${VBENCH_ROOT}/evaluate.py" >&2; exit 1; }
fi

command=(
  "${VBENCH_PYTHON}"
  "${PROJECT_ROOT}/paper/aaai27/experiments/run_vbench_factorials.py"
  "${ACTION}"
  --factorial-root "${FACTORIAL_ROOT}"
  --dimension temporal_flickering
  --raw-subdir vbench_raw_temporal_flickering
  --output-name vbench_temporal_flickering.json
  --ngpus 4
  --python "${VBENCH_PYTHON}"
)
if [[ "${ACTION}" == "run" ]]; then
  command+=(--vbench-root "${VBENCH_ROOT}")
fi

echo "P1 temporal_flickering: action=${ACTION}; root=${FACTORIAL_ROOT}; GPUs=${GPU_IDS}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${command[@]}"

if [[ "${ACTION}" != "prepare" ]]; then
  "${VBENCH_PYTHON}" \
    "${PROJECT_ROOT}/paper/aaai27/experiments/compile_vbench_paired_statistics.py" \
    --factorial-root "${FACTORIAL_ROOT}" \
    --vbench-json "${FACTORIAL_ROOT}/metrics/vbench_temporal_flickering.json" \
    --dimension temporal_flickering \
    --output "${FACTORIAL_ROOT}/metrics/vbench_temporal_flickering_paired_statistics.csv"
fi
