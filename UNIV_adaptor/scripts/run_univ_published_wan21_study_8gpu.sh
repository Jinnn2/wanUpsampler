#!/usr/bin/env bash
# New output only; reuse native Wan and the previously validated RGB/HR helpers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
PILOT_VENV="${PILOT_VENV:-${REPO_ROOT}/.venv_published_wan21}"
DEFAULT_PYTHON="${BASE_PYTHON}"
[[ ! -x "${PILOT_VENV}/bin/python" ]] || DEFAULT_PYTHON="${PILOT_VENV}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
# Deliberately ignore inherited DATASET_ROOT/PUBLISHED_WAN21_ROOT from old runs.
STUDY_ROOT="${PUBLISHED_WAN21_STUDY_ROOT:-${REPO_ROOT}/outputs/published_wan21_study_v2}"
STUDY_CONFIG="${STUDY_CONFIG:-${REPO_ROOT}/UNIV_adaptor/configs/published_wan21_study_v2.json}"
SOURCE_PUBLISHED_ROOT="${SOURCE_PUBLISHED_ROOT:-${REPO_ROOT}/outputs/published_wan21_pilot_v3}"
SR_CHECKPOINT="${SR_CHECKPOINT:-/mnt/afs_2/houze/models/hy15_endpoint_v1/RealESRGAN_x2plus.pth}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490}"
NGPUS="${NGPUS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/published_wan21_study.py"
COMMON=(--out "${STUDY_ROOT}" --config "${STUDY_CONFIG}" --model-root "${MODEL_ROOT}" --sr-checkpoint "${SR_CHECKPOINT}" --python "${PYTHON_BIN}" --ngpus "${NGPUS}" --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" --expected-vbench-commit "${EXPECTED_VBENCH_COMMIT}")
[[ "${SOURCE_PUBLISHED_ROOT}" == "none" ]] || COMMON+=(--reuse-calibration-root "${SOURCE_PUBLISHED_ROOT}")
cd "${REPO_ROOT}"
case "${MODE}" in
  setup-sr)
    # Use the existing published generation venv. Avoid replacing PyTorch,
    # torchvision, CUDA, diffusers, numpy or the scoring environment.
    [[ -x "${PILOT_VENV}/bin/python" ]] || { echo 'Run the ORIGINAL published pilot setup first.' >&2; exit 1; }
    "${PILOT_VENV}/bin/python" -m pip install --no-deps realesrgan==0.3.0 basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8
    "${PILOT_VENV}/bin/python" -m pip install --no-deps addict==2.4.0 future==1.0.0 lmdb==1.6.2 yapf==0.43.0 filterpy==1.4.5
    "${PILOT_VENV}/bin/python" -c 'from changing_resolution_distill.realesrgan_compat import install_functional_tensor_shim; install_functional_tensor_shim(); from realesrgan import RealESRGANer; print("SR imports OK; run check for checkpoint/CUDA smoke test")'
    ;;
  plan|reuse-calibration|check|calibrate|audit|generate|status|finalize|score|report|blind|blind-report|export|diagnose)
    shift || true
    "${PYTHON_BIN}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  blind-score|serve)
    HUMAN_MODE="score"
    [[ "${MODE}" != "serve" ]] || HUMAN_MODE="serve"
    "${VBENCH_PYTHON}" UNIV_adaptor/scripts/data/acceleration_blind_audit.py "${HUMAN_MODE}" --out "${STUDY_ROOT}/blind" --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" --expected-vbench-commit "${EXPECTED_VBENCH_COMMIT}" --ngpus "${NGPUS}"
    ;;
  help)
    echo 'Modes: setup-sr plan reuse-calibration check calibrate audit generate status finalize score report blind blind-score blind-report export serve diagnose'
    echo 'Reuse existing Wan weights/SR checkpoint. New root only; old v3 calibration files are verified and reused by reference.'
    echo 'Required: check -> calibrate -> audit -> generate -> finalize -> score. See UNIV_adaptor/PUBLISHED_WAN21_STUDY.md.'
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
