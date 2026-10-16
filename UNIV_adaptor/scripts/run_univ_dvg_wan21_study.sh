#!/usr/bin/env bash
# Official DVG vs FULL on Wan2.1-T2V-1.3B (832x480x81), then VBench and requirement scoring.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
GEN_PYTHON="${GEN_PYTHON:-${REPO_ROOT}/.venv_published_wan21/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
DVG_ROOT="${DVG_ROOT:-/mnt/afs_2/houze/DVG}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PROMPTS="${VBENCH_PROMPTS:-${VBENCH_ROOT}/prompts/all_dimension.txt}"
PROMPTS="${PROMPTS:-${REPO_ROOT}/prompts/univ_controlled_factor_v1.jsonl}"
STUDY_ROOT="${STUDY_ROOT:-${REPO_ROOT}/outputs/dvg_wan21_study_v1}"
REQ_ROOT="${REQ_ROOT:-${REPO_ROOT}/outputs/dvg_wan21_requirements_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/dvg_wan21_study.py"
REQ="${REPO_ROOT}/UNIV_adaptor/scripts/run_univ_controlled_factor_requirements.sh"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"

case "${MODE}" in
  plan)
    "${BASE_PYTHON}" "${DRIVER}" plan --out "${STUDY_ROOT}" --prompts "${PROMPTS}" --vbench-prompts "${VBENCH_PROMPTS}"
    ;;
  smoke)
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${GEN_PYTHON}" "${DRIVER}" smoke --out "${STUDY_ROOT}" \
      --dvg-root "${DVG_ROOT}" --ckpt-dir "${MODEL_ROOT}"
    ;;
  generate)
    mkdir -p "${STUDY_ROOT}/logs"
    pids=()
    for i in "${!GPUS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${GEN_PYTHON}" "${DRIVER}" generate --out "${STUDY_ROOT}" \
        --dvg-root "${DVG_ROOT}" --ckpt-dir "${MODEL_ROOT}" --shard-index "${i}" --shard-count "${#GPUS[@]}" \
        > "${STUDY_ROOT}/logs/generate_${i}.log" 2>&1 &
      pids+=("$!")
    done
    trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
    failed=0
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "worker ${i} failed: ${STUDY_ROOT}/logs/generate_${i}.log" >&2; failed=1; }
    done
    exit "${failed}"
    ;;
  status)
    "${BASE_PYTHON}" "${DRIVER}" status --out "${STUDY_ROOT}"
    ;;
  score)
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VBENCH_PYTHON}" "${DRIVER}" score --out "${STUDY_ROOT}" \
      --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" --ngpus "${#GPUS[@]}"
    ;;
  requirements)
    "${BASE_PYTHON}" "${REPO_ROOT}/UNIV_adaptor/scripts/data/score_controlled_factor_requirements.py" plan \
      --dvg-study "${STUDY_ROOT}" --checklist "${REPO_ROOT}/prompts/univ_controlled_factor_v1_requirements.jsonl" \
      --out "${REQ_ROOT}"
    REQ_ROOT="${REQ_ROOT}" QUALITY_CSV="${STUDY_ROOT}/quality_by_video.csv" bash "${REQ}" signals
    REQ_ROOT="${REQ_ROOT}" QUALITY_CSV="${STUDY_ROOT}/quality_by_video.csv" bash "${REQ}" vlm
    REQ_ROOT="${REQ_ROOT}" QUALITY_CSV="${STUDY_ROOT}/quality_by_video.csv" bash "${REQ}" report
    ;;
  export)
    tar czf "${STUDY_ROOT}_analysis.tgz" -C "$(dirname "${STUDY_ROOT}")" \
      "$(basename "${STUDY_ROOT}")/plan.json" "$(basename "${STUDY_ROOT}")/receipts" \
      "$(basename "${STUDY_ROOT}")/quality_by_video.csv" "$(basename "${STUDY_ROOT}")/repro_by_video.csv" \
      "$(basename "${STUDY_ROOT}")/repro_summary.md" "$(basename "${STUDY_ROOT}")/vbench_scores.json"
    echo "Download: ${STUDY_ROOT}_analysis.tgz and ${REQ_ROOT}_analysis.tgz"
    ;;
  *)
    echo "Usage: bash $0 {plan|smoke|generate|status|score|requirements|export}"
    echo "Clone DVG first: git clone https://github.com/DarrenZheng303/DVG.git ${DVG_ROOT}"
    echo "Plan: 64 dev prompts x 3 seeds x {FULL, DVG 0.7, DVG 0.5} + 48 VBench prompts x {FULL, DVG 0.5}."
    ;;
esac
