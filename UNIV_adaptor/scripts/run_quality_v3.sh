#!/usr/bin/env bash
# Quality layer v3 (QUALITY_V3.md): the three-way pairwise judge of quality_tri.py on every accelerated clip of the
# attention-combined quality plan against its same-seed FULL, plus the three FULL seed pairs of every prompt (null).
# Then the dev -> dev2 router is rerun with quality_tri in place of quality_kept.
#   bash UNIV_adaptor/scripts/run_quality_v3.sh <stage>     # plan judge export route pack | all | status
# The judge is resumable (one cache file per pair); rerun `judge` after an interruption.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
STAGE="${1:-all}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
Q_ROOT="${Q_ROOT:-${REPO_ROOT}/outputs/quality_attn}"
FIXED="${FIXED:-${REPO_ROOT}/outputs/fixed_metrics_attn}"
DYN_CSV="${DYN_CSV:-${REPO_ROOT}/outputs/dynamic_attn/dynamic_by_video.csv}"
OLD_ROUTE="${OLD_ROUTE:-${REPO_ROOT}/outputs/route_attn}"
OUT="${V3_ROOT:-${REPO_ROOT}/outputs/quality_v3}"
ROUTE="${OUT}/route"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
D="${REPO_ROOT}/UNIV_adaptor/scripts/data/quality_tri.py"
S="${REPO_ROOT}/UNIV_adaptor/scripts"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
say() { printf '\n######## %s  (%s)\n' "$*" "$(date '+%F %T')"; }

plan() {
  say "plan"
  [[ -f "${OUT}/plan.json" ]] || "${BASE_PYTHON}" "${D}" plan-all --quality-root "${Q_ROOT}" --out "${OUT}"
  mkdir -p "${OUT}/pair" "${OUT}/logs"
  local r
  for r in outputs/quality_tri outputs/quality_tri_seeds; do  # pilot answers: same protocol, same cache key
    [[ -d "${r}/pair" ]] && cp -n "${r}"/pair/*.json "${OUT}/pair/" 2>/dev/null || true
  done
  echo "cached judge calls: $(ls "${OUT}/pair" | wc -l)"
}

judge() {
  say "judge on GPUs ${GPU_IDS}"
  local pids=() i p failed=0
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${VLM_PYTHON}" -u "${D}" pair --out "${OUT}" --model "${JUDGE_MODEL}" \
      --shard-index "${i}" --shard-count "${#GPUS[@]}" >> "${OUT}/logs/pair_${i}.log" 2>&1 &
    pids+=("$!")
  done
  trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
  for p in "${pids[@]}"; do wait "${p}" || failed=1; done
  (( failed == 0 )) || { echo "a worker failed; see ${OUT}/logs"; tail -n 5 "${OUT}"/logs/*.log; exit 1; }
}

export_() {
  say "export"
  "${BASE_PYTHON}" "${D}" export --out "${OUT}" | tee "${OUT}/report.txt"
}

route() {
  say "route: dev -> dev2 with quality_tri"
  mkdir -p "${ROUTE}"
  [[ -f "${OLD_ROUTE}/cells_llm.json" ]] && cp -n "${OLD_ROUTE}/cells_llm.json" "${ROUTE}/"
  local m
  for m in table evaluate; do
    FIXED_ROOT="${FIXED}" DYN_CSV="${DYN_CSV}" Q_CSV="${OUT}/quality_tri_by_video.csv" Q_COL=quality_tri \
      ROUTE_ROOT="${ROUTE}" bash "${S}/run_univ_route.sh" "${m}"
  done
}

pack() {
  local files=(plan.json quality_tri_by_video.csv tri_by_pair.csv report.txt) keep=() f
  for f in route/route_table.csv route/table_inputs.json route/route_eval.csv route/route_rules.json \
           route/route_report.md route/evaluate.log; do files+=("${f}"); done
  for f in "${files[@]}"; do [[ -e "${OUT}/${f}" ]] && keep+=("$(basename "${OUT}")/${f}"); done
  tar czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "${keep[@]}"
  echo "SEND THIS: ${OUT}.tgz ($(du -h "${OUT}.tgz" | cut -f1))"
}

status() {
  local total done_
  total=$("${BASE_PYTHON}" -c "import json,sys; print(len({r['key'] for r in json.load(open(sys.argv[1]))['pairs']}))" "${OUT}/plan.json")
  done_=$(ls "${OUT}/pair" | wc -l)
  echo "judged ${done_} / ${total} (cache may also hold pilot-only pairs)"
  tail -n 1 "${OUT}"/logs/pair_*.log 2>/dev/null || true
}

case "${STAGE}" in
  plan) plan ;;
  judge) judge ;;
  export) export_ ;;
  route) route ;;
  pack) pack ;;
  status) status ;;
  all) plan && judge && export_ && route && pack ;;
  *) echo "unknown stage ${STAGE}"; exit 2 ;;
esac
