#!/usr/bin/env bash
# Router v1 (route_v1.py): fastest prompt -> method policy that keeps tau of FULL's semantics and visual quality,
# with the cell predicted from the prompt text. No generation; needs the finished dynamic and quality layers.
#   bash UNIV_adaptor/scripts/run_univ_route.sh table      # join speed, retention, quality_kept (CPU)
#   bash UNIV_adaptor/scripts/run_univ_route.sh classify   # zero-shot text -> cell with Qwen3-VL (one GPU, minutes)
#   bash UNIV_adaptor/scripts/run_univ_route.sh evaluate   # CV + dev->dev2, deployment rules, report (CPU)
#   bash UNIV_adaptor/scripts/run_univ_route.sh all
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
FIXED_ROOT="${FIXED_ROOT:-${REPO_ROOT}/outputs/fixed_metrics_v1}"
DYN_CSV="${DYN_CSV:-${REPO_ROOT}/outputs/dynamic_v1_all/dynamic_by_video.csv}"
Q_CSV="${Q_CSV:-${REPO_ROOT}/outputs/quality_v1/quality_by_video.csv}"
ROUTE_ROOT="${ROUTE_ROOT:-${REPO_ROOT}/outputs/route_v1}"
GPU="${GPU:-0}"
export PYTHONDONTWRITEBYTECODE=1
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/route_v1.py"
cd "${REPO_ROOT}"

case "${MODE}" in
  table)
    "${BASE_PYTHON}" "${DRIVER}" table --fixed-root "${FIXED_ROOT}" --dynamic "${DYN_CSV}" --quality "${Q_CSV}" --out "${ROUTE_ROOT}"
    ;;
  classify)
    CUDA_VISIBLE_DEVICES="${GPU}" "${VLM_PYTHON}" "${DRIVER}" classify --model "${JUDGE_MODEL}" --out "${ROUTE_ROOT}"
    ;;
  evaluate)
    "${BASE_PYTHON}" "${DRIVER}" evaluate --out "${ROUTE_ROOT}" | tee "${ROUTE_ROOT}/evaluate.log"
    files=(route_table.csv table_inputs.json route_eval.csv route_rules.json route_report.md)
    [[ -f "${ROUTE_ROOT}/cells_llm.json" ]] && files+=(cells_llm.json)
    tar czf "${ROUTE_ROOT}_analysis.tgz" -C "${ROUTE_ROOT}" "${files[@]}"
    echo "Download: ${ROUTE_ROOT}_analysis.tgz"
    ;;
  all)
    bash "$0" table && bash "$0" classify && bash "$0" evaluate
    ;;
  *)
    sed -n '2,8p' "$0"
    exit 1
    ;;
esac
