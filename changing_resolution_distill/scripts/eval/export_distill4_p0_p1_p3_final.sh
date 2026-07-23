#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
MAIN_ROOT="${DISTILL4_FINAL_QUALITY_EFFICIENCY:-${PROJECT_ROOT}/outputs/aaai27_experiments/quality_efficiency_distill4}"
P3_ROOT="${TALH_VALIDATION_ROOT:-${PROJECT_ROOT}/outputs/aaai27_experiments/distill4_talh_validation_sweep}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
EXPORT_ROOT="${DISTILL4_EXPORT_ROOT:-${PROJECT_ROOT}/exports/distill4_p0_p1_p3_final_${timestamp}}"
INCLUDE_VIDEOS="${INCLUDE_VIDEOS:-0}"

command=(
  "${WAN_PYTHON}"
  "${PROJECT_ROOT}/paper/aaai27/experiments/export_distill4_p0_p1_p3.py"
  --project-root "${PROJECT_ROOT}"
  --main-root "${MAIN_ROOT}"
  --validation-root "${P3_ROOT}"
  --output-root "${EXPORT_ROOT}"
)
if [[ "${INCLUDE_VIDEOS}" == "1" ]]; then
  command+=(--include-videos)
fi

echo "Exporting completed Distill4 P0/P1/P3 results"
echo "Main suite : ${MAIN_ROOT}"
echo "P3 sweep   : ${P3_ROOT}"
echo "Destination: ${EXPORT_ROOT}.tar.gz"
"${command[@]}"

archive="${EXPORT_ROOT}.tar.gz"
[[ -f "${archive}" ]] || { echo "Archive was not created: ${archive}" >&2; exit 1; }
sha256sum "${archive}" > "${archive}.sha256"
echo "Archive SHA256: $(cut -d' ' -f1 "${archive}.sha256")"
