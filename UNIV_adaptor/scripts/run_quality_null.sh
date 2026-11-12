#!/usr/bin/env bash
# Null controls for the pairwise quality layer (quality_null.py): identity pairs, FULL-vs-FULL across seeds, and the
# reverse question (FULL >= arm) for SVG2, SPARGE50, SCALING10 and TEA008. Reuses the quality layer's judge, protocol
# and forward cache, so only the new ordered pairs are judged.
#   bash UNIV_adaptor/scripts/run_quality_null.sh          # plan, judge (GPUs) + PSNR (CPU), report, bundle
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
Q_ROOT="${Q_ROOT:-${REPO_ROOT}/outputs/quality_attn}"
OUT="${NULL_ROOT:-${REPO_ROOT}/outputs/quality_null}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PSNR_SHARDS="${PSNR_SHARDS:-8}"
D="${REPO_ROOT}/UNIV_adaptor/scripts/data/quality_null.py"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
say() { printf '\n######## %s  (%s)\n' "$*" "$(date '+%F %T')"; }

say "plan"
[[ -f "${OUT}/plan.json" ]] || "${BASE_PYTHON}" "${D}" plan --quality-root "${Q_ROOT}" --out "${OUT}"
mkdir -p "${OUT}/pair" "${OUT}/logs"
cp -n "${Q_ROOT}"/pair/*.json "${OUT}/pair/" 2>/dev/null || true  # forward (arm >= FULL) answers already judged
echo "seeded $(ls "${OUT}/pair" | wc -l) cached judge calls from ${Q_ROOT}/pair"

say "judge on GPUs ${GPU_IDS}; PSNR on ${PSNR_SHARDS} CPU shards alongside"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${VLM_PYTHON}" "${D}" pair --out "${OUT}" --model "${JUDGE_MODEL}" \
    --shard-index "${i}" --shard-count "${#GPUS[@]}" > "${OUT}/logs/pair_${i}.log" 2>&1 &
  pids+=("$!")
done
for ((i = 0; i < PSNR_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES="" "${VLM_PYTHON}" "${D}" psnr --out "${OUT}" --shard-index "${i}" --shard-count "${PSNR_SHARDS}" \
    > "${OUT}/logs/psnr_${i}.log" 2>&1 &
  pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
failed=0
for p in "${pids[@]}"; do wait "${p}" || failed=1; done
(( failed == 0 )) || { echo "a worker failed; see ${OUT}/logs"; tail -n 5 "${OUT}"/logs/*.log; exit 1; }

say "report"
"${BASE_PYTHON}" "${D}" report --out "${OUT}" | tee "${OUT}/report.txt"
tar czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "$(basename "${OUT}")/plan.json" "$(basename "${OUT}")/null_by_pair.csv" \
  "$(basename "${OUT}")/report.txt"
echo "SEND THIS: ${OUT}.tgz ($(du -h "${OUT}.tgz" | cut -f1))"
