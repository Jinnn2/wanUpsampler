#!/usr/bin/env bash
# Pilot of a three-way pairwise quality question (A better / B better / Same), quality_tri.py: identity, FULL seed
# 100 vs 42, and six arms vs FULL, on 32 prompts (16 high + 16 low motion), seed 42. Same judge and frames as the
# quality layer; the old binary answers (quality_null, quality_attn caches) are joined for comparison.
#   bash UNIV_adaptor/scripts/run_quality_tri.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
Q_ROOT="${Q_ROOT:-${REPO_ROOT}/outputs/quality_attn}"
NULL_ROOT="${NULL_ROOT:-${REPO_ROOT}/outputs/quality_null}"
OUT="${TRI_ROOT:-${REPO_ROOT}/outputs/quality_tri}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PROMPTS="${PROMPTS:-16}"
D="${REPO_ROOT}/UNIV_adaptor/scripts/data/quality_tri.py"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
say() { printf '\n######## %s  (%s)\n' "$*" "$(date '+%F %T')"; }

say "plan"
[[ -f "${OUT}/plan.json" ]] || "${BASE_PYTHON}" "${D}" plan --quality-root "${Q_ROOT}" --out "${OUT}" --prompts "${PROMPTS}"
mkdir -p "${OUT}/logs"

say "judge on GPUs ${GPU_IDS}"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${VLM_PYTHON}" -u "${D}" pair --out "${OUT}" --model "${JUDGE_MODEL}" \
    --shard-index "${i}" --shard-count "${#GPUS[@]}" > "${OUT}/logs/pair_${i}.log" 2>&1 &
  pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
failed=0
for p in "${pids[@]}"; do wait "${p}" || failed=1; done
(( failed == 0 )) || { echo "a worker failed; see ${OUT}/logs"; tail -n 5 "${OUT}"/logs/*.log; exit 1; }

say "report"
"${BASE_PYTHON}" "${D}" report --out "${OUT}" --old-cache "${NULL_ROOT}/pair" "${Q_ROOT}/pair" | tee "${OUT}/report.txt"
tar czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "$(basename "${OUT}")/plan.json" "$(basename "${OUT}")/tri_by_pair.csv" \
  "$(basename "${OUT}")/report.txt"
echo "SEND THIS: ${OUT}.tgz ($(du -h "${OUT}.tgz" | cut -f1))"
