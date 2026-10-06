#!/usr/bin/env bash
# Independent environment/output. Does NOT run or alter the old generation study.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
NATIVE_PYTHON="${NATIVE_PYTHON:-${REPO_ROOT}/.venv_published_wan21/bin/python}"
FLASH_VENV="${FLASH_VENV:-${REPO_ROOT}/.venv_flashvsr_diagnostic}"
FLASH_PYTHON="${FLASH_PYTHON:-${FLASH_VENV}/bin/python}"
FLASH_ROOT="${FLASH_ROOT:-${REPO_ROOT}/UNIV_adaptor/external/flashvsr}"
FLASH_KERNEL_ROOT="${FLASH_KERNEL_ROOT:-${REPO_ROOT}/UNIV_adaptor/external/flashvsr_block_sparse}"
FLASH_WEIGHTS="${FLASH_WEIGHTS:-${REPO_ROOT}/checkpoints/flashvsr_v11}"
FLASH_DIAG_ROOT="${FLASH_DIAG_ROOT:-${REPO_ROOT}/outputs/flashvsr_asset_diagnostic_v1}"
SOURCE_STUDY_ROOT="${SOURCE_STUDY_ROOT:-${REPO_ROOT}/outputs/published_wan21_study_v2}"
NGPUS="${NGPUS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/flashvsr_asset_diagnostic.py"
COMMON=(--out "${FLASH_DIAG_ROOT}" --study-root "${SOURCE_STUDY_ROOT}" --flash-root "${FLASH_ROOT}" --kernel-root "${FLASH_KERNEL_ROOT}" --weights "${FLASH_WEIGHTS}" --ngpus "${NGPUS}")
cd "${REPO_ROOT}"
case "${MODE}" in
  fetch)
    "${BASE_PYTHON}" "${DRIVER}" fetch "${COMMON[@]}" "$@"
    ;;
  setup)
    "${BASE_PYTHON}" "${DRIVER}" fetch "${COMMON[@]}"
    # Never reuse/update the old Wan or VBench environment.
    "${BASE_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3,11), "Set BASE_PYTHON to Python 3.11 for the author environment"'
    [[ "${FLASH_VENV}" != "${REPO_ROOT}/.venv_published_wan21" ]] || { echo 'Refusing to replace native generation environment' >&2; exit 1; }
    if [[ ! -x "${FLASH_PYTHON}" ]]; then
      "${BASE_PYTHON}" -m venv "${FLASH_VENV}"
    fi
    "${FLASH_PYTHON}" -c 'import sys; from pathlib import Path; actual=Path(sys.prefix).resolve(); expected=Path(sys.argv[1]).resolve(); native=Path(sys.argv[2]).parent.parent.resolve(); assert sys.prefix != sys.base_prefix and actual == expected and actual != native, "Refusing to install into a global/native or unexpected Python environment"' "${FLASH_VENV}" "${NATIVE_PYTHON}"
    "${FLASH_PYTHON}" -m pip install --upgrade pip
    "${FLASH_PYTHON}" -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
    "${FLASH_PYTHON}" -m pip install -r "${FLASH_ROOT}/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu124
    "${FLASH_PYTHON}" -m pip install packaging ninja wheel setuptools
    # pip builds local CUDA packages in-place. Build a separate checkout so
    # generated build/egg-info files cannot dirty the audited author source.
    mkdir -p "${REPO_ROOT}/outputs"
    FLASH_BUILD_DIR="$(mktemp -d "${REPO_ROOT}/outputs/flashvsr_kernel_build.XXXXXX")"
    FLASH_KERNEL_COMMIT="$(git -C "${FLASH_KERNEL_ROOT}" rev-parse HEAD)"
    git clone --no-checkout "${FLASH_KERNEL_ROOT}" "${FLASH_BUILD_DIR}/kernel"
    git -C "${FLASH_BUILD_DIR}/kernel" checkout --detach "${FLASH_KERNEL_COMMIT}"
    git -C "${FLASH_BUILD_DIR}/kernel" submodule update --init --recursive
    "${FLASH_PYTHON}" -m pip install --no-build-isolation --no-deps "${FLASH_BUILD_DIR}/kernel"
    echo "Build checkout retained for diagnosis: ${FLASH_BUILD_DIR}"
    "${FLASH_PYTHON}" -c 'import torch; from block_sparse_attn import block_sparse_attn_func; print("Sparse backend import OK:", torch.__version__, torch.version.cuda)'
    # Source is imported from the pinned checkout; no editable-install metadata
    # is written into author repositories, preserving checkout audits.
    "${FLASH_PYTHON}" "${DRIVER}" download "${COMMON[@]}" "$@"
    ;;
  download)
    "${FLASH_PYTHON}" "${DRIVER}" download "${COMMON[@]}" "$@"
    ;;
  plan|status|report|export)
    "${FLASH_PYTHON}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  prepare)
    # VAE decoding only. No DiT, T5, RealESRGAN or HR4.
    [[ -x "${NATIVE_PYTHON}" ]] || { echo "Native generation Python missing: ${NATIVE_PYTHON}" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="${PREPARE_GPU:-0}" "${NATIVE_PYTHON}" "${DRIVER}" prepare "${COMMON[@]}" "$@"
    ;;
  check)
    CUDA_VISIBLE_DEVICES="${CHECK_GPU:-0}" "${FLASH_PYTHON}" "${DRIVER}" check "${COMMON[@]}" "$@"
    ;;
  run)
    "${FLASH_PYTHON}" "${DRIVER}" run "${COMMON[@]}" --sr-python "${FLASH_PYTHON}" "$@"
    ;;
  help)
    echo 'Modes: fetch setup download plan prepare check run status report export'
    echo 'First run: setup -> plan -> prepare -> check -> run -> report -> export'
    echo 'Default: 4 preselected prompts x 2 seeds x S_B025; 16 SR jobs, first 33 frames.'
    echo 'Overrides on plan only: --prompt-ids p00 --seeds 3407 --frames 33 --spatial-arms S_B025'
    echo 'SOURCE_STUDY_ROOT and FLASH_DIAG_ROOT are independent of inherited DATASET_ROOT.'
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
