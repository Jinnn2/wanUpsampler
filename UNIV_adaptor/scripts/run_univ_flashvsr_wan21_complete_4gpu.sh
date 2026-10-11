#!/usr/bin/env bash
# Full-length native LR -> FlashVSR completion. Reuse existing environments/assets.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
NATIVE_PYTHON="${NATIVE_PYTHON:-${REPO_ROOT}/.venv_published_wan21/bin/python}"
FLASH_PYTHON="${FLASH_PYTHON:-${REPO_ROOT}/.venv_flashvsr_diagnostic/bin/python}"
SOURCE_STUDY_ROOT="${SOURCE_STUDY_ROOT:-${REPO_ROOT}/outputs/published_wan21_study_v2}"
FLASH_COMPLETE_ROOT="${FLASH_COMPLETE_ROOT:-${REPO_ROOT}/outputs/flashvsr_wan21_complete_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/complete_flashvsr_wan21.py"
COMMON=(--out "${FLASH_COMPLETE_ROOT}" --study-root "${SOURCE_STUDY_ROOT}"
  --flash-root "${FLASH_ROOT:-${REPO_ROOT}/UNIV_adaptor/external/flashvsr}"
  --kernel-root "${FLASH_KERNEL_ROOT:-${REPO_ROOT}/UNIV_adaptor/external/flashvsr_block_sparse}"
  --weights "${FLASH_WEIGHTS:-${REPO_ROOT}/checkpoints/flashvsr_v11}"
  --native-python "${NATIVE_PYTHON}" --sr-python "${FLASH_PYTHON}" --gpu-ids "${GPU_IDS}")
cd "${REPO_ROOT}"
case "${MODE}" in
  all|check|prepare|run)
    for interpreter in "${BASE_PYTHON}" "${NATIVE_PYTHON}" "${FLASH_PYTHON}"; do
      [[ -x "${interpreter}" ]] || { echo "Missing existing interpreter: ${interpreter}; reuse the working diagnostic setup." >&2; exit 1; }
    done
    "${BASE_PYTHON}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  plan|status|finalize|export)
    "${BASE_PYTHON}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  *)
    echo "Usage: bash $0 {all|plan|check|prepare|run|status|finalize|export}"
    echo "Defaults: four GPUs 0,1,2,3; 24 clips, 81 frames, 832x480, 16 fps."
    echo "Example: GPU_IDS=4,5,6,7 bash $0 all"
    echo "No setup/download, new DiT generation, HR4, or HR-downsample control."
    ;;
esac
