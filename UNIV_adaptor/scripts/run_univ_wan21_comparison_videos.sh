#!/usr/bin/env bash
# CPU presentation only. No video generation or VBench rerun.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/make_wan21_comparison_videos.py"
COMMON=(--score-root "${FLASH_SCORE_ROOT:-${REPO_ROOT}/outputs/flashvsr_wan21_complete_score_v1}"
  --out "${COMPARISON_ROOT:-${REPO_ROOT}/outputs/wan21_five_method_comparisons_v1}"
  --panel-width "${PANEL_WIDTH:-832}" --jobs "${JOBS:-2}" --threads "${ENCODE_THREADS:-2}"
  --ffmpeg "${FFMPEG:-ffmpeg}")
[[ -z "${FONT_FILE:-}" ]] || COMMON+=(--font "${FONT_FILE}")
[[ -z "${FFPROBE:-}" ]] || COMMON+=(--ffprobe "${FFPROBE}")
cd "${REPO_ROOT}"
case "${MODE}" in
  all|plan|render|index|export)
    "${BASE_PYTHON}" "${DRIVER}" "${MODE}" "${COMMON[@]}" "$@"
    ;;
  *)
    echo "Usage: bash $0 {all|plan|render|index|export}"
    echo "24 score-labelled five-panel videos, native 832x480 per panel by default."
    echo "CPU-only FFmpeg/Pillow; JOBS=2, ENCODE_THREADS=2. Originals untouched."
    echo "Optional: --group-ids p02_s3407 (use a separate COMPARISON_ROOT for subsets)."
    [[ "${MODE}" == help ]] || exit 2
    ;;
esac
