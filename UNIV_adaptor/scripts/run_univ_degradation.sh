#!/usr/bin/env bash
# Synthetic-degradation control (synthetic_degradation.py): FULL clips degraded by a known kind and amount, scored by
# every layer with its unchanged code. Ground truth is the construction, so no human labels are needed.
# Needs the fixed-layer root of the main study (for the FULL clips), .venv_vlm3 and the VBench checkout.
#   bash UNIV_adaptor/scripts/run_univ_degradation.sh make      # plan -> degrade (CPU) -> fixed plan
#   bash UNIV_adaptor/scripts/run_univ_degradation.sh all       # make -> fixed -> dynamic -> quality -> vbench -> report
#   bash UNIV_adaptor/scripts/run_univ_degradation.sh report    # any time; missing layers are left empty
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
SRC_FIXED_ROOT="${SRC_FIXED_ROOT:-${REPO_ROOT}/outputs/fixed_metrics_v1}"
DEG_ROOT="${DEG_ROOT:-${REPO_ROOT}/outputs/degradation_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CPU_JOBS="${CPU_JOBS:-16}"
FFMPEG="${FFMPEG:-ffmpeg}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/synthetic_degradation.py"
SCRIPTS="${REPO_ROOT}/UNIV_adaptor/scripts"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"
# The layer runners take their roots from the environment; all of them read the degradation fixed plan.
export FIXED_ROOT="${DEG_ROOT}/fixed" DYN_ROOT="${DEG_ROOT}/dynamic" Q_ROOT="${DEG_ROOT}/quality" GPU_IDS

case "${MODE}" in
  plan)
    "${BASE_PYTHON}" "${DRIVER}" plan --fixed-root "${SRC_FIXED_ROOT}" --out "${DEG_ROOT}"
    ;;
  make)
    [[ -f "${DEG_ROOT}/plan.json" ]] || bash "$0" plan
    mkdir -p "${DEG_ROOT}/logs"
    pids=()
    for ((i = 0; i < CPU_JOBS; i++)); do
      "${BASE_PYTHON}" "${DRIVER}" make --out "${DEG_ROOT}" --ffmpeg "${FFMPEG}" --shard-index "${i}" \
        --shard-count "${CPU_JOBS}" > "${DEG_ROOT}/logs/make_${i}.log" 2>&1 &
      pids+=("$!")
    done
    trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
    failed=0
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "make worker ${i} failed: ${DEG_ROOT}/logs/make_${i}.log" >&2; failed=1; }
    done
    [[ "${failed}" == 0 ]] || exit 1
    "${BASE_PYTHON}" "${DRIVER}" finalize --out "${DEG_ROOT}"
    ;;
  fixed)
    bash "${SCRIPTS}/run_univ_fixed_metrics.sh" signals && bash "${SCRIPTS}/run_univ_fixed_metrics.sh" report
    ;;
  dynamic)
    [[ -f "${DYN_ROOT}/plan.json" ]] || bash "${SCRIPTS}/run_univ_dynamic.sh" plan all
    bash "${SCRIPTS}/run_univ_dynamic.sh" run all
    ;;
  quality)
    [[ -f "${Q_ROOT}/plan.json" ]] || bash "${SCRIPTS}/run_univ_quality.sh" plan
    bash "${SCRIPTS}/run_univ_quality.sh" run
    ;;
  vbench)
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${BASE_PYTHON}" "${DRIVER}" vbench --out "${DEG_ROOT}" --ngpus "${#GPUS[@]}" \
      2>&1 | tee "${DEG_ROOT}/logs/vbench.log"
    ;;
  report)
    "${BASE_PYTHON}" "${DRIVER}" report --out "${DEG_ROOT}"
    files=(plan.json degradation_report.md degradation_by_video.csv)
    for f in fixed/fixed_by_video.csv dynamic/dynamic_by_video.csv quality/quality_by_video.csv vbench_by_video.csv; do
      [[ -f "${DEG_ROOT}/${f}" ]] && files+=("${f}")
    done
    tar czf "${DEG_ROOT}_analysis.tgz" -C "${DEG_ROOT}" "${files[@]}"
    echo "Download: ${DEG_ROOT}_analysis.tgz"
    ;;
  status)
    echo "videos: $(ls "${DEG_ROOT}/videos" 2>/dev/null | grep -vc tmp) / $((64 * 19))"
    echo "fixed cache: $(ls "${FIXED_ROOT}/cache" 2>/dev/null | wc -l)"
    echo "dynamic vlm: $(ls "${DYN_ROOT}/vlm" 2>/dev/null | wc -l)  pair: $(ls "${DYN_ROOT}/pair" 2>/dev/null | wc -l)"
    echo "quality vlm: $(ls "${Q_ROOT}/vlm" 2>/dev/null | wc -l)  pair: $(ls "${Q_ROOT}/pair" 2>/dev/null | wc -l)"
    echo "vbench: $([[ -f "${DEG_ROOT}/vbench_by_video.csv" ]] && echo done || echo pending)"
    ;;
  all)
    bash "$0" make && bash "$0" fixed && bash "$0" quality && bash "$0" dynamic && bash "$0" vbench && bash "$0" report
    ;;
  *)
    sed -n '2,7p' "$0"
    exit 1
    ;;
esac
