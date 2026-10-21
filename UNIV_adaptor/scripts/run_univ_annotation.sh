#!/usr/bin/env bash
# M3 of REQUIREMENTS_V2_SPEC.md: human annotation of the dynamic layer. Needs the finished full dynamic run.
#   bash UNIV_adaptor/scripts/run_univ_annotation.sh sample    # pre-registered sample, once (commit sample.json)
#   bash UNIV_adaptor/scripts/run_univ_annotation.sh kit       # offline kits for annotators A and B (.tgz)
#   bash UNIV_adaptor/scripts/run_univ_annotation.sh analyze labels_A.json labels_B.json
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
DYN_ROOT="${DYN_ROOT:-${REPO_ROOT}/outputs/dynamic_v1_all}"
ANN_ROOT="${ANN_ROOT:-${REPO_ROOT}/outputs/annotation_v1}"
FFMPEG="${FFMPEG:-ffmpeg}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/annotation_v2.py"
export PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"

case "${MODE}" in
  sample)
    "${BASE_PYTHON}" "${DRIVER}" sample --dyn-root "${DYN_ROOT}" --out "${ANN_ROOT}"
    # The sample goes into the repo before anyone labels (pre-registration).
    mkdir -p prompts/annotation_v1 && cp "${ANN_ROOT}/sample.json" prompts/annotation_v1/sample.json
    sha256sum prompts/annotation_v1/sample.json
    echo "Before sending kits: git add -f prompts/annotation_v1/sample.json && git commit -m \"M3: pre-registered sample\" && git push"
    ;;
  kit)
    for who in A B; do
      "${BASE_PYTHON}" "${DRIVER}" kit --out "${ANN_ROOT}" --annotator "${who}" --ffmpeg "${FFMPEG}" | tail -n 1
      tar czf "${ANN_ROOT}/kit_${who}.tgz" -C "${ANN_ROOT}" "kit_${who}"
      echo "Send: ${ANN_ROOT}/kit_${who}.tgz ($(du -h "${ANN_ROOT}/kit_${who}.tgz" | cut -f1))"
    done
    ;;
  analyze)
    shift
    "${BASE_PYTHON}" "${DRIVER}" analyze --out "${ANN_ROOT}" --labels "$@"
    echo "Report: ${ANN_ROOT}/analysis.md"
    ;;
  *)
    echo "Usage: bash $0 {sample|kit|analyze labels_A.json labels_B.json}"
    echo "Dynamic results: ${DYN_ROOT}; annotation root: ${ANN_ROOT}"
    ;;
esac
