#!/usr/bin/env bash
# Fixed-layer metrics (REQUIREMENTS_V2_SPEC.md, layer 1) over every finished development run. No generation.
#   bash UNIV_adaptor/scripts/run_univ_fixed_metrics.sh all      # plan -> check -> signals -> report
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
SIGNAL_PYTHON="${SIGNAL_PYTHON:-${BASE_PYTHON}}"
OUT="${REPO_ROOT}/outputs"
FIXED_ROOT="${FIXED_ROOT:-${OUT}/fixed_metrics_v1}"
PUBLISHED="${PUBLISHED:-${OUT}/published_wan21_platform_v1 ${OUT}/published_wan21_twostage_v1 ${OUT}/published_wan21_twostage_v2b ${OUT}/published_wan21_platform_dev2 ${OUT}/published_wan21_twostage_dev2}"
DVG_STUDIES="${DVG_STUDIES:-${OUT}/dvg_wan21_study_v1 ${OUT}/dvg_wan21_study_dev2}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/score_fixed_metrics.py"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"

case "${MODE}" in
  plan)
    # shellcheck disable=SC2086
    "${BASE_PYTHON}" "${DRIVER}" plan --published ${PUBLISHED} --dvg-study ${DVG_STUDIES} --out "${FIXED_ROOT}"
    ;;
  check)
    # A few clips on one GPU first, so a broken environment fails in a minute rather than on 8 GPUs.
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${SIGNAL_PYTHON}" "${DRIVER}" signals --out "${FIXED_ROOT}" \
      --shard-index 0 --shard-count 2000
    cached=("${FIXED_ROOT}"/cache/*.json)
    "${BASE_PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); d.pop("flow_mag_series"); print(json.dumps(d, indent=1))' "${cached[0]}"
    ;;
  signals)
    mkdir -p "${FIXED_ROOT}/logs"
    pids=()
    for i in "${!GPUS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${SIGNAL_PYTHON}" "${DRIVER}" signals --out "${FIXED_ROOT}" \
        --shard-index "${i}" --shard-count "${#GPUS[@]}" > "${FIXED_ROOT}/logs/signals_${i}.log" 2>&1 &
      pids+=("$!")
    done
    trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
    failed=0
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "worker ${i} failed: ${FIXED_ROOT}/logs/signals_${i}.log" >&2; failed=1; }
    done
    exit "${failed}"
    ;;
  report)
    "${BASE_PYTHON}" "${DRIVER}" report --out "${FIXED_ROOT}" | tee "${FIXED_ROOT}/report.txt"
    tar czf "${FIXED_ROOT}_analysis.tgz" -C "$(dirname "${FIXED_ROOT}")" \
      "$(basename "${FIXED_ROOT}")/plan.json" "$(basename "${FIXED_ROOT}")/cache" \
      "$(basename "${FIXED_ROOT}")/fixed_by_video.csv" "$(basename "${FIXED_ROOT}")/report.txt"
    echo "Download: ${FIXED_ROOT}_analysis.tgz"
    ;;
  status)
    echo "cache: $(ls "${FIXED_ROOT}/cache" 2>/dev/null | wc -l) clips"
    ;;
  all)
    bash "$0" plan && bash "$0" check && bash "$0" signals && bash "$0" report
    ;;
  *)
    echo "Usage: bash $0 {plan|check|signals|report|status|all}"
    echo "Sources: ${PUBLISHED} ${DVG_STUDIES}"
    ;;
esac
