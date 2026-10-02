#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
SOURCE_PUBLISHED_ROOT="${SOURCE_PUBLISHED_ROOT:-${REPO_ROOT}/outputs/published_wan21_pilot_v3}"
JENGA_DIAGNOSTIC_ROOT="${JENGA_DIAGNOSTIC_ROOT:-${REPO_ROOT}/outputs/published_wan21_jenga_diagnostic_v1}"
PILOT_VENV="${PILOT_VENV:-${REPO_ROOT}/.venv_published_wan21}"
if [[ -x "${PILOT_VENV}/bin/python" ]]; then
  DEFAULT_PYTHON="${PILOT_VENV}/bin/python"
else
  DEFAULT_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
export PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"
case "${MODE}" in
  plan|check|generate|report)
    shift || true
    "${PYTHON_BIN}" UNIV_adaptor/scripts/data/published_wan21_jenga_diagnostic.py "${MODE}" \
      --source "${SOURCE_PUBLISHED_ROOT}" --out "${JENGA_DIAGNOSTIC_ROOT}" \
      --ngpus "${NGPUS:-8}" --python "${PYTHON_BIN}" "$@"
    ;;
  help)
    echo 'Modes: plan check generate report. Six diagnostic videos, default GPUs 0/1/2, three excluded full-shape warmups.'
    echo 'Reuses frozen v3 calibration read-only. Does NOT release or alter the original pilot.'
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
