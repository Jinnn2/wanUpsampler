#!/usr/bin/env bash
# Requirement-aware scoring of controlled-factor development videos. No generation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
SIGNAL_PYTHON="${SIGNAL_PYTHON:-${BASE_PYTHON}}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm/bin/python}"
export VLM_MODEL="${VLM_MODEL:-/mnt/afs_2/houze/models/Qwen2.5-VL-7B-Instruct}"
CF_ROOT="${CF_ROOT:-${REPO_ROOT}/outputs/univ_controlled_factor_v1}"
REQ_ROOT="${REQ_ROOT:-${REPO_ROOT}/outputs/controlled_factor_requirements_v1}"
CHECKLIST="${CHECKLIST:-${REPO_ROOT}/prompts/univ_controlled_factor_v1_requirements.jsonl}"
# PUBLISHED_ROOT: score a finalized + VBench-scored published_wan21 platform instead of CF_ROOT;
# plan then writes the converted VBench CSV into REQ_ROOT.
PUBLISHED_ROOT="${PUBLISHED_ROOT:-}"
if [[ -n "${PUBLISHED_ROOT}" ]]; then
  QUALITY_CSV="${QUALITY_CSV:-${REQ_ROOT}/quality_by_video.csv}"
else
  QUALITY_CSV="${QUALITY_CSV:-${CF_ROOT}/metrics/controlled_factor_vbench/quality_by_video.csv}"
fi
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/score_controlled_factor_requirements.py"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"

fan_out() {  # $1 = python, $2 = mode; one worker per listed GPU, fails if any worker fails
  local python="$1" mode="$2" pids=() i
  mkdir -p "${REQ_ROOT}/logs"
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${python}" "${DRIVER}" "${mode}" --out "${REQ_ROOT}" \
      --shard-index "${i}" --shard-count "${#GPUS[@]}" > "${REQ_ROOT}/logs/${mode}_${i}.log" 2>&1 &
    pids+=("$!")
  done
  trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
  local failed=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "worker ${i} failed: ${REQ_ROOT}/logs/${mode}_${i}.log" >&2; failed=1; }
  done
  return "${failed}"
}

case "${MODE}" in
  setup-vlm)
    [[ -x "${VLM_PYTHON}" ]] || "${BASE_PYTHON}" -m venv --system-site-packages "${REPO_ROOT}/.venv_vlm"
    "${VLM_PYTHON}" -m pip install "transformers==4.51.3" "accelerate>=0.34" av
    [[ -d "${VLM_MODEL}" ]] || "${VLM_PYTHON}" -m huggingface_hub.commands.huggingface_cli download \
      Qwen/Qwen2.5-VL-7B-Instruct --local-dir "${VLM_MODEL}"
    ;;
  plan)
    if [[ -n "${PUBLISHED_ROOT}" ]]; then
      "${BASE_PYTHON}" "${DRIVER}" plan --published "${PUBLISHED_ROOT}" --checklist "${CHECKLIST}" --out "${REQ_ROOT}"
    else
      "${BASE_PYTHON}" "${DRIVER}" plan --root "${CF_ROOT}" --checklist "${CHECKLIST}" --out "${REQ_ROOT}"
    fi
    ;;
  check)
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${VLM_PYTHON}" "${DRIVER}" check --out "${REQ_ROOT}"
    ;;
  signals) fan_out "${SIGNAL_PYTHON}" signals ;;
  vlm) fan_out "${VLM_PYTHON}" vlm ;;
  report)
    "${BASE_PYTHON}" "${DRIVER}" report --out "${REQ_ROOT}" --quality-csv "${QUALITY_CSV}" | tee "${REQ_ROOT}/report.txt"
    tar czf "${REQ_ROOT}_analysis.tgz" -C "$(dirname "${REQ_ROOT}")" \
      "$(basename "${REQ_ROOT}")/plan.json" "$(basename "${REQ_ROOT}")/signals" "$(basename "${REQ_ROOT}")/vlm" \
      "$(basename "${REQ_ROOT}")/analysis" "$(basename "${REQ_ROOT}")/report.txt"
    echo "Download: ${REQ_ROOT}_analysis.tgz"
    ;;
  all)
    bash "$0" plan && bash "$0" check && bash "$0" signals && bash "$0" vlm && bash "$0" report
    ;;
  status)
    for d in signals vlm; do echo "${d}: $(ls "${REQ_ROOT}/${d}" 2>/dev/null | wc -l) receipts"; done
    ;;
  *)
    echo "Usage: bash $0 {setup-vlm|plan|check|signals|vlm|report|all|status}"
    echo "Default GPUs 0,1,2,3 (any count works). 768 development videos; locked test never decoded."
    ;;
esac
