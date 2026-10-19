#!/usr/bin/env bash
# Two-stage spatial-only / temporal-only platform (LR81+FlashVSR, HR41+RIFE, controls).
# Install once per machine with install_twostage_wan21_env.sh; it records the
# working interpreters in outputs/twostage_env/<gpu-tag>.env, which is sourced here.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
GPU_TAG="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
ENV_FILE="${TWOSTAGE_ENV_FILE:-${REPO_ROOT}/outputs/twostage_env/${GPU_TAG}.env}"
# shellcheck disable=SC1090
[[ -f "${ENV_FILE}" ]] && source "${ENV_FILE}"
WAN_PYTHON="${WAN_PYTHON:-${REPO_ROOT}/.venv_published_wan21/bin/python}"
FLASH_PYTHON="${FLASH_PYTHON:-${REPO_ROOT}/.venv_flashvsr_diagnostic/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
TWOSTAGE_ROOT="${TWOSTAGE_ROOT:-${REPO_ROOT}/outputs/published_wan21_twostage_v1}"
NGPUS="${NGPUS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/twostage_wan21.py"
COMMON=(--out "${TWOSTAGE_ROOT}" --model-root "${MODEL_ROOT}" --ngpus "${NGPUS}"
  --python "${WAN_PYTHON}" --flash-python "${FLASH_PYTHON}"
  --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}")
cd "${REPO_ROOT}"
# The driver itself is stdlib-only; like platform v1 it runs in the Wan venv.
run() { "${WAN_PYTHON}" "${DRIVER}" "$1" "${COMMON[@]}" "${@:2}"; }
case "${MODE}" in
  plan|check|stage1|post|status|finalize|report|export)
    run "${MODE}" "$@"
    ;;
  score)
    # VBench itself runs in VBENCH_PYTHON.
    run score "$@"
    run export
    ;;
  vbench-smoke)
    run vbench-smoke --smoke-video "${SMOKE_VIDEO:?Set SMOKE_VIDEO to any existing 832x480 MP4}" "$@"
    ;;
  all)
    # Stops at the first failure (set -e); rerunning resumes from valid receipts.
    run check
    run stage1
    run post
    run finalize
    run score
    run export
    ;;
  help)
    echo 'Modes: check stage1 post status finalize score report export all vbench-smoke'
    echo 'First run: bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh, then this script with "all".'
    echo "Output: ${TWOSTAGE_ROOT}; env file: ${ENV_FILE}"
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
