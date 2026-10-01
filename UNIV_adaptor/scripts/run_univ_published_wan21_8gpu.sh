#!/usr/bin/env bash
# Published pipelines; one persistent single-GPU worker per arm per GPU.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
PILOT_VENV="${PILOT_VENV:-${REPO_ROOT}/.venv_published_wan21}"
if [[ -x "${PILOT_VENV}/bin/python" ]]; then
  DEFAULT_PYTHON="${PILOT_VENV}/bin/python"
else
  DEFAULT_PYTHON="${BASE_PYTHON}"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
if [[ -n "${PUBLISHED_WAN21_ROOT:-}" ]]; then
  DATASET_ROOT="${PUBLISHED_WAN21_ROOT}"
elif [[ -n "${DATASET_ROOT:-}" ]]; then
  echo "Warning: using inherited DATASET_ROOT=${DATASET_ROOT}. Prefer PUBLISHED_WAN21_ROOT for this experiment." >&2
else
  DATASET_ROOT="${REPO_ROOT}/outputs/published_wan21_pilot_v1"
fi
export PYTHONDONTWRITEBYTECODE=1
PILOT_CONFIG="${PILOT_CONFIG:-${REPO_ROOT}/UNIV_adaptor/configs/published_wan21_pilot_v1.json}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490}"
NGPUS="${NGPUS:-8}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/published_wan21_pilot.py"
COMMON=(--config "${PILOT_CONFIG}" --out "${DATASET_ROOT}" --model-root "${MODEL_ROOT}" --python "${PYTHON_BIN}" --ngpus "${NGPUS}" --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" --expected-vbench-commit "${EXPECTED_VBENCH_COMMIT}")
cd "${REPO_ROOT}"
case "${MODE}" in
  setup)
    # Keep CUDA/PyTorch binaries from the working generation environment;
    # isolate Python-only dependencies, never pip-upgrade the scoring env.
    "${BASE_PYTHON}" -m venv --system-site-packages "${PILOT_VENV}"
    "${PILOT_VENV}/bin/python" -m pip install -r UNIV_adaptor/configs/published_wan21_requirements.txt
    if ! "${PILOT_VENV}/bin/python" -c 'import flash_attn' >/dev/null 2>&1; then
      if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
        "${PILOT_VENV}/bin/python" -m pip install flash-attn==2.7.4.post1 --no-build-isolation
      else
        echo 'Missing flash-attn. Use a compatible existing CUDA/PyTorch environment, or rerun setup with INSTALL_FLASH_ATTN=1.' >&2
        exit 1
      fi
    fi
    echo "Isolated generation Python: ${PILOT_VENV}/bin/python"
    ;;
  fetch|check|plan|calibrate|audit|generate|status|finalize|score|report|blind|blind-report|export|diagnose)
    shift || true
    "${PYTHON_BIN}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  blind-score|serve)
    HUMAN_MODE="score"
    [[ "${MODE}" == "serve" ]] && HUMAN_MODE="serve"
    "${VBENCH_PYTHON}" UNIV_adaptor/scripts/data/acceleration_blind_audit.py "${HUMAN_MODE}" --out "${DATASET_ROOT}/blind" --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" --expected-vbench-commit "${EXPECTED_VBENCH_COMMIT}" --ngpus "${NGPUS}"
    ;;
  help)
    echo 'Modes: fetch setup check plan calibrate audit generate status finalize score report blind blind-score blind-report export serve diagnose'
    echo 'Run calibrate + audit before generate. Existing Wan2.1-1.3B weights are reused; no automatic weight download.'
    echo 'See UNIV_adaptor/PUBLISHED_WAN21_PILOT.md. Ctrl+C stops active generation workers; validated records are resumable.'
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
