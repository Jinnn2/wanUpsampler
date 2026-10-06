#!/usr/bin/env bash
# Evaluation only. No setup, model downloads, rebuilding kernels or generation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
SOURCE_FLASH_DIAG_ROOT="${SOURCE_FLASH_DIAG_ROOT:-${REPO_ROOT}/outputs/flashvsr_asset_diagnostic_v1}"
FLASH_EVAL_ROOT="${FLASH_EVAL_ROOT:-${REPO_ROOT}/outputs/flashvsr_spatial_baseline_eval_v1}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
NGPUS="${NGPUS:-8}"
export PYTHONDONTWRITEBYTECODE=1
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/evaluate_flashvsr_spatial_baseline.py"
COMMON=(--source "${SOURCE_FLASH_DIAG_ROOT}" --out "${FLASH_EVAL_ROOT}" --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" --ngpus "${NGPUS}")
cd "${REPO_ROOT}"
case "${MODE}" in
  all)
    for phase in plan check blind export-rater score report export-analysis; do
      "${BASE_PYTHON}" "${DRIVER}" "${phase}" "${COMMON[@]}" "$@"
    done
    ;;
  plan|check|blind|score|report|human-report|export-rater|export-analysis)
    "${BASE_PYTHON}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  *)
    echo "Usage: bash $0 {all|plan|check|blind|score|report|human-report|export-rater|export-analysis} [driver options]"
    echo "Source: ${SOURCE_FLASH_DIAG_ROOT}"
    echo "Evaluation: ${FLASH_EVAL_ROOT} (independent; source assets remain untouched)"
    echo "No setup or generation. VBench uses 8 GPUs per dimension, sequentially."
    [[ "${MODE}" == help ]] || exit 2
    ;;
esac
