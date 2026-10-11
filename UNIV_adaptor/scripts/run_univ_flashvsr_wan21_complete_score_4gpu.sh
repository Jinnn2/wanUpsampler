#!/usr/bin/env bash
# Score only the 24 NEW full-length Flash clips; reuse 96 old strict scores.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/score_flashvsr_wan21_complete.py"
COMMON=(--source "${FLASH_COMPLETE_ROOT:-${REPO_ROOT}/outputs/flashvsr_wan21_complete_v1}"
  --out "${FLASH_SCORE_ROOT:-${REPO_ROOT}/outputs/flashvsr_wan21_complete_score_v1}"
  --vbench-root "${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
  --vbench-python "${VBENCH_PYTHON:-/opt/conda/bin/python}" --gpu-ids "${GPU_IDS:-0,1,2,3}")
cd "${REPO_ROOT}"
case "${MODE}" in
  all|plan|check|score|report|export)
    "${BASE_PYTHON}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  *)
    echo "Usage: bash $0 {all|plan|check|score|report|export}"
    echo "Four GPUs; 24 Flash clips x 7 dimensions. Reuse old 96 full-video scores."
    echo "Example: GPU_IDS=4,5,6,7 bash $0 all"
    echo "No generation/setup; independent score output and small analysis archive."
    [[ "${MODE}" == help ]] || exit 2
    ;;
esac
