#!/usr/bin/env bash
# Attention v1 (UNIV_adaptor/ATTENTION_V1.md): SVG2 + SPARGE50 on the platform v1 + dev2 prompts (3 seeds), on the
# platform's official-Wan path (attention_wan.py), scored with the three layers against the platform FULL50.
# Needs the H100 machine that made the platform FULL50 (the gate checks bitwise reuse first).
#   bash UNIV_adaptor/scripts/run_attention_v1.sh            # all stages, unattended; ends with SEND THIS
#   bash UNIV_adaptor/scripts/run_attention_v1.sh <stage>    # gate gen finalize vbench fixed dynamic quality route pack status
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
STAGE="${1:-all}"
OUT="${REPO_ROOT}/outputs"
S="${REPO_ROOT}/UNIV_adaptor/scripts"
A="${S}/data/attention_wan.py"
PY="/opt/conda/bin/python -u"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
ARMS=(SVG2 SPARGE50)
SETS=(v1 dev2)
FIXED="${OUT}/fixed_metrics_attn"
DYN="${OUT}/dynamic_attn"
Q="${OUT}/quality_attn"
ROUTE="${OUT}/route_attn"
# Every source of the dev analyses (run_univ_fixed_metrics.sh defaults) plus the two attention roots, so routing sees
# old and new arms together. Old clips are not recomputed: the caches are seeded from the dev roots (keyed by sha256).
OLD_PUBLISHED="${OUT}/published_wan21_platform_v1 ${OUT}/published_wan21_twostage_v1 ${OUT}/published_wan21_twostage_v2b ${OUT}/published_wan21_platform_dev2 ${OUT}/published_wan21_twostage_dev2"
OLD_DVG="${OUT}/dvg_wan21_study_v1 ${OUT}/dvg_wan21_study_dev2"
say() { printf '\n######## %s  (%s)\n' "$*" "$(date '+%F %T')"; }
cfg() { echo "${REPO_ROOT}/UNIV_adaptor/configs/published_wan21_platform_$1.json"; }
root() { echo "${OUT}/attention_wan_$1"; }

TAG="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
WAN_PYTHON=""
# shellcheck disable=SC1090
[[ -f "${OUT}/twostage_env/${TAG}.env" ]] && source "${OUT}/twostage_env/${TAG}.env"
export PYTHON_WAN21="${PYTHON_WAN21:-${WAN_PYTHON:-${REPO_ROOT}/.venv_published_wan21/bin/python}}"

seed_cache() {  # seed_cache FROM TO: copy cached per-clip results without overwriting
  [[ -d "$1" ]] || { echo "no cache to seed from: $1"; return 0; }
  mkdir -p "$2"
  cp -rn "$1"/. "$2"/
  echo "seeded $(ls "$2" | wc -l) cached results into $2 from $1"
}

gate() {
  local s
  for s in "${SETS[@]}"; do
    say "gate ${s}: DENSE on 2 prompts, seed 42, must equal the platform FULL50 bitwise"
    ${PY} "${A}" run --config "$(cfg "${s}")" --out "$(root "${s}")" --arms DENSE --first 2 --seeds 42 --gpus "${GPU_IDS}"
    ${PY} "${A}" gate --out "$(root "${s}")" --reference "${OUT}/published_wan21_platform_${s}" | tee "$(root "${s}")/gate.txt"
    grep -q "GATE PASSED" "$(root "${s}")/gate.txt" || { echo "gate failed for ${s}; stop"; exit 1; }
  done
}

gen() {
  local s try
  for s in "${SETS[@]}"; do
    for try in 1 2 3; do  # a hung SpargeAttn kernel is killed by the watchdog; its jobs are retried here
      say "generate ${s} (${ARMS[*]}), attempt ${try}"
      ${PY} "${A}" run --config "$(cfg "${s}")" --out "$(root "${s}")" --arms "${ARMS[@]}" --gpus "${GPU_IDS}" && break
    done
  done
}

finalize() {
  local s
  for s in "${SETS[@]}"; do
    ${PY} "${A}" finalize --config "$(cfg "${s}")" --out "$(root "${s}")" --reference "${OUT}/published_wan21_platform_${s}" --arms "${ARMS[@]}"
    ${PY} "${A}" report --out "$(root "${s}")"
  done
}

vbench() {
  local s
  for s in "${SETS[@]}"; do
    say "VBench ${s}"
    ${PY} "${A}" score --out "$(root "${s}")" --ngpus "$(tr ',' '\n' <<< "${GPU_IDS}" | wc -l)"
  done
}

fixed() {
  say "fixed layer"
  seed_cache "${OUT}/fixed_metrics_v1/cache" "${FIXED}/cache"
  FIXED_ROOT="${FIXED}" PUBLISHED="${OLD_PUBLISHED} $(root v1) $(root dev2)" DVG_STUDIES="${OLD_DVG}" GPU_IDS="${GPU_IDS}" \
    bash "${S}/run_univ_fixed_metrics.sh" all
}

dynamic() {
  say "dynamic layer (retention)"
  seed_cache "${OUT}/dynamic_v1_all/vlm" "${DYN}/vlm"
  seed_cache "${OUT}/dynamic_v1_all/pair" "${DYN}/pair"
  local m
  for m in plan run; do
    FIXED_ROOT="${FIXED}" DYN_ROOT="${DYN}" GPU_IDS="${GPU_IDS}" bash "${S}/run_univ_dynamic.sh" "${m}" all
  done
}

quality() {
  say "quality layer + v2 judder/flicker checks"
  seed_cache "${OUT}/quality_v1/vlm" "${Q}/vlm"
  seed_cache "${OUT}/quality_v1/pair" "${Q}/pair"
  local m
  for m in plan run v2; do
    FIXED_ROOT="${FIXED}" Q_ROOT="${Q}" GPU_IDS="${GPU_IDS}" CALIB_FIXED="${OUT}/fixed_metrics_v1/fixed_by_video.csv" \
      bash "${S}/run_univ_quality.sh" "${m}"
  done
}

route() {
  say "routing table + dev -> dev2 evaluation"
  local m
  for m in table evaluate; do
    FIXED_ROOT="${FIXED}" DYN_CSV="${DYN}/dynamic_by_video.csv" Q_CSV="${Q}_v2/quality_by_video.csv" ROUTE_ROOT="${ROUTE}" \
      bash "${S}/run_univ_route.sh" "${m}"
  done
}

pack() {
  local files=() s
  for s in "${SETS[@]}"; do
    files+=("attention_wan_${s}/dataset_manifest.json" "attention_wan_${s}/plan.json" "attention_wan_${s}/report.txt"
            "attention_wan_${s}/gate.txt" "attention_wan_${s}/metrics/quality_by_video.csv" "attention_wan_${s}/logs")
  done
  files+=(fixed_metrics_attn/fixed_by_video.csv fixed_metrics_attn/report.txt fixed_metrics_attn/plan.json
          dynamic_attn/dynamic_by_video.csv dynamic_attn/report.txt quality_attn/quality_by_video.csv quality_attn/report.txt
          quality_attn_v2/quality_by_video.csv quality_attn_v2/report.txt route_attn_analysis.tgz attention_v1.log)
  local keep=()
  for f in "${files[@]}"; do [[ -e "${OUT}/${f}" ]] && keep+=("${f}"); done
  tar czf "${OUT}/attention_v1_results.tgz" -C "${OUT}" "${keep[@]}"
  echo "SEND THIS: ${OUT}/attention_v1_results.tgz ($(du -h "${OUT}/attention_v1_results.tgz" | cut -f1))"
}

status() {
  local s
  for s in "${SETS[@]}"; do
    echo "${s}: $(ls "$(root "${s}")/records" 2>/dev/null | wc -l) receipts; manifest $([[ -f "$(root "${s}")/dataset_manifest.json" ]] && echo yes || echo no); VBench $([[ -f "$(root "${s}")/metrics/quality_by_video.csv" ]] && echo yes || echo no)"
  done
  echo "fixed:   $(ls "${FIXED}/cache" 2>/dev/null | wc -l) clips cached"
  echo "dynamic: vlm $(ls "${DYN}/vlm" 2>/dev/null | wc -l), pair $(ls "${DYN}/pair" 2>/dev/null | wc -l)"
  echo "quality: vlm $(ls "${Q}/vlm" 2>/dev/null | wc -l), pair $(ls "${Q}/pair" 2>/dev/null | wc -l)"
  ls "${ROUTE}_analysis.tgz" 2>/dev/null || true
}

case "${STAGE}" in
  all)
    echo "GPU ${TAG}; worker python ${PYTHON_WAN21}"
    for st in gate gen finalize vbench fixed dynamic quality route; do
      bash "$0" "${st}" || { say "STAGE FAILED: ${st}"; status; exit 1; }
    done
    status
    pack
    say "ATTENTION V1 DONE" ;;
  gate|gen|finalize|vbench|fixed|dynamic|quality|route|pack|status) "${STAGE}" ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
