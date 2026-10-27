#!/usr/bin/env bash
# Representative clips for the thesis figures (figure_samples.py): fixed selection rules over the finished fixed,
# dynamic and quality v2 layers, then frame strips, x-t slices and side-by-side videos with FFmpeg. CPU only.
#   bash UNIV_adaptor/scripts/run_univ_figure_samples.sh pins     # DVG short-prompt collapse pins from the DVG repro set
#   bash UNIV_adaptor/scripts/run_univ_figure_samples.sh select   # manifest.json + candidates.csv (seconds)
#   bash UNIV_adaptor/scripts/run_univ_figure_samples.sh render   # strips / slices / compare.mp4 / index.html
#   bash UNIV_adaptor/scripts/run_univ_figure_samples.sh pack     # tgz to download
#   bash UNIV_adaptor/scripts/run_univ_figure_samples.sh all      # pins -> select -> render -> pack
# select adds PINS (default: the file pins writes, when present); PINS= (empty) leaves the pins out.
# ARMS_B="STEP20 TEA015 ..." and METRIC=min|avg|geo set figure B; PANELS="A1_freeze B_hh" renders a subset.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
FFMPEG="${FFMPEG:-ffmpeg}"
FIXED_ROOT="${FIXED_ROOT:-${REPO_ROOT}/outputs/fixed_metrics_v1}"
DYN_CSV="${DYN_CSV:-${REPO_ROOT}/outputs/dynamic_v1_all/dynamic_by_video.csv}"
Q_CSV="${Q_CSV:-${REPO_ROOT}/outputs/quality_v1_v2/quality_by_video.csv}"
DVG_STUDY="${DVG_STUDY:-${REPO_ROOT}/outputs/dvg_wan21_study_v1}"
FIG_ROOT="${FIG_ROOT:-${REPO_ROOT}/outputs/figure_samples_v1}"
PINS="${PINS-${FIG_ROOT}/pins.json}"
ARMS_B="${ARMS_B:-STEP20 TEA015 RCM4 F41_RIFE LR_FLASH}"
METRIC="${METRIC:-min}"
export PYTHONDONTWRITEBYTECODE=1
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/figure_samples.py"
cd "${REPO_ROOT}"

case "${MODE}" in
  pins)
    "${BASE_PYTHON}" "${DRIVER}" pins-dvg --dvg-study "${DVG_STUDY}" --out "${FIG_ROOT}/pins.json"
    ;;
  select)
    mkdir -p "${FIG_ROOT}"
    extra=()
    [[ -n "${PINS}" && -f "${PINS}" ]] && extra+=(--pins "${PINS}")
    # shellcheck disable=SC2086
    "${BASE_PYTHON}" "${DRIVER}" select --fixed-root "${FIXED_ROOT}" --dynamic "${DYN_CSV}" --quality "${Q_CSV}" \
      --arms-b ${ARMS_B} --metric "${METRIC}" --out "${FIG_ROOT}" "${extra[@]}" | tee "${FIG_ROOT}/select.log"
    ;;
  render)
    extra=()
    # shellcheck disable=SC2206
    [[ -n "${PANELS:-}" ]] && extra+=(--panels ${PANELS})
    "${BASE_PYTHON}" "${DRIVER}" render --out "${FIG_ROOT}" --ffmpeg "${FFMPEG}" "${extra[@]}"
    ;;
  pack)
    tar czf "${FIG_ROOT}.tgz" -C "$(dirname "${FIG_ROOT}")" "$(basename "${FIG_ROOT}")"
    echo "Download: ${FIG_ROOT}.tgz (open render/index.html)"
    ;;
  all)
    bash "$0" pins && bash "$0" select && bash "$0" render && bash "$0" pack
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
