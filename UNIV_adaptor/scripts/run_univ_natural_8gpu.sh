#!/usr/bin/env bash
# Natural prompt set v1 (natural_prompts.py): 256 real VidProM user prompts x 1 seed x 8 frontier arms on one 8-GPU
# machine, scored by the same three layers, then prompt-vector routing vs blind / predicted cells.
# Two steps, because the prompt set is reviewed and committed before any GPU time is spent on it:
#   bash UNIV_adaptor/scripts/run_univ_natural_8gpu.sh prompts   # ~20 min: download, filter, Qwen screen, draw 256
#        -> send outputs/natural_v1_prompts.tgz; the prompts and configs come back as a commit; git pull
#   bash UNIV_adaptor/scripts/run_univ_natural_8gpu.sh all       # ~9 h: everything below, in this order
# Stages (each resumes from its own receipts): checklist platform twostage fixed dynamic quality route; status.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-help}"
OUT="${REPO_ROOT}/outputs"
S="${REPO_ROOT}/UNIV_adaptor/scripts"
CFG="${REPO_ROOT}/UNIV_adaptor/configs"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
PROMPTS="${REPO_ROOT}/prompts/univ_natural_v1.jsonl"
CONTROLLED=("${REPO_ROOT}/prompts/univ_controlled_factor_v1.jsonl" "${REPO_ROOT}/prompts/univ_controlled_factor_dev2.jsonl")
SEL="${OUT}/natural_v1_prompts"
P="${OUT}/published_wan21_platform_natural"
T="${OUT}/published_wan21_twostage_natural"
GEN="${OUT}/requirements_natural"
CHECKLIST="${GEN}/checklist.jsonl"
FIXED="${OUT}/fixed_metrics_natural"
DYN="${OUT}/dynamic_natural"
Q="${OUT}/quality_natural"
ROUTE="${OUT}/route_natural"
ALL_GPUS="0,1,2,3,4,5,6,7"
IFS=',' read -r -a GPUS <<< "${ALL_GPUS}"
export NGPUS=8 PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"
say() { printf '\n######## %s  (%s)\n' "$*" "$(date '+%F %T')"; }
need() { [[ -f "$1" ]] || { echo "missing $1 ($2)" >&2; exit 1; }; }

gen_env() {  # interpreters verified by install_twostage_wan21_env.sh, as in the dev2 batch
  local tag env_file
  tag="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
  env_file="${TWOSTAGE_ENV_FILE:-${OUT}/twostage_env/${tag}.env}"
  need "${env_file}" "run install_twostage_wan21_env.sh first"
  # shellcheck disable=SC1090
  source "${env_file}"
  export TWOSTAGE_ENV_FILE="${env_file}" PYTHON_BIN="${WAN_PYTHON}" GEN_PYTHON="${WAN_PYTHON}"
  export VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
}

prompts() {
  local d="${REPO_ROOT}/UNIV_adaptor/scripts/data/natural_prompts.py"
  say "prompts: VidProM candidates"
  [[ -f "${SEL}/candidates.jsonl" ]] || "${VLM_PYTHON}" "${d}" candidates --out "${SEL}"
  say "prompts: Qwen screen (one GPU)"
  [[ -f "${SEL}/screen.json" ]] || CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${VLM_PYTHON}" "${d}" screen --out "${SEL}" --model "${JUDGE_MODEL}"
  "${BASE_PYTHON}" "${d}" select --out "${SEL}" | tee "${SEL}/select.txt"
  "${BASE_PYTHON}" "${d}" configs --prompts "${SEL}/univ_natural_v1.jsonl" --config-dir "${SEL}"
  tar czf "${SEL}.tgz" -C "${OUT}" "$(basename "${SEL}")/univ_natural_v1.jsonl" "$(basename "${SEL}")/select.txt" \
    "$(basename "${SEL}")/candidates.jsonl" "$(basename "${SEL}")/candidates_meta.json" "$(basename "${SEL}")/screen.json" \
    "$(basename "${SEL}")"/published_wan21_*_natural.json
  echo "SEND THIS: ${SEL}.tgz"
}

checklist() {
  say "checklist: dynamic-layer questions for the natural prompts (Qwen3-VL-32B, machine only)"
  need "${PROMPTS}" "run the prompts stage, send the tgz, git pull"
  local ids pids=() i
  mapfile -t ids < <("${BASE_PYTHON}" -c 'import json,sys; [print(json.loads(l)["prompt_id"]) for l in open(sys.argv[1]) if l.strip()]' "${PROMPTS}")
  mkdir -p "${GEN}/logs"
  for i in "${!GPUS[@]}"; do
    # Few-shot examples are the pilot's watchmaking family, as for v2.0; its prompts live in the controlled files.
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${VLM_PYTHON}" "${S}/data/requirements_v2.py" generate \
      --prompts "${PROMPTS}" "${CONTROLLED[@]}" --examples "${REPO_ROOT}/prompts/univ_requirements_v2_pilot.jsonl" \
      --model "${JUDGE_MODEL}" --out "${GEN}" --only "${ids[@]}" --shard-index "${i}" --shard-count "${#GPUS[@]}" \
      > "${GEN}/logs/generate_${i}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "worker ${i} failed: ${GEN}/logs/generate_${i}.log" >&2; failed=1; }; done
  (( failed == 0 )) || return 1
  # Prompts whose checklist still fails validation are listed and left out of the dynamic layer.
  "${BASE_PYTHON}" "${S}/data/requirements_v2.py" collect --prompts "${PROMPTS}" --out "${GEN}" | tee "${GEN}/collect.txt"
}

platform() {
  say "platform natural: FULL50 STEP20 TEA015 SCALING10"
  gen_env
  local m
  for m in check calibrate generate finalize score export; do
    say "platform ${m}"
    PILOT_CONFIG="${CFG}/published_wan21_platform_natural.json" PUBLISHED_WAN21_ROOT="${P}" \
      bash "${S}/run_univ_published_wan21_8gpu.sh" "${m}"
  done
}

twostage() {
  say "two-stage natural: LR_FLASH F41_RIFE RCM4 LR41_FLASH_RIFE (FULL50 from platform)"
  gen_env
  bash "${S}/install_twostage_wan21_env.sh" rcm
  need "${P}/dataset_manifest.json" "platform natural must be finalized first"
  TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_natural.json" TWOSTAGE_ROOT="${T}" REFERENCE_ROOT="${P}" \
    bash "${S}/run_univ_twostage_wan21_8gpu.sh" all
}

fixed() {
  say "fixed layer"
  FIXED_ROOT="${FIXED}" PUBLISHED="${P} ${T}" DVG_STUDIES="" GPU_IDS="${ALL_GPUS}" bash "${S}/run_univ_fixed_metrics.sh" all
}

dynamic() {
  say "dynamic layer (retention)"
  need "${CHECKLIST}" "checklist stage"
  local m
  for m in plan run; do
    FIXED_ROOT="${FIXED}" CHECKLIST="${CHECKLIST}" PROMPT_FILES="${PROMPTS}" DYN_ROOT="${DYN}" GPU_IDS="${ALL_GPUS}" \
      bash "${S}/run_univ_dynamic.sh" "${m}" all
  done
}

quality() {
  say "quality layer + v2 judder/flicker checks (thresholds from the main experiment's FULL clips)"
  local m
  for m in plan run v2; do
    FIXED_ROOT="${FIXED}" Q_ROOT="${Q}" GPU_IDS="${ALL_GPUS}" CALIB_FIXED="${OUT}/fixed_metrics_v1/fixed_by_video.csv" \
      bash "${S}/run_univ_quality.sh" "${m}"
  done
}

route() {
  say "routing: table, prompt vectors, CV comparison"
  local env=(FIXED_ROOT="${FIXED}" DYN_CSV="${DYN}/dynamic_by_video.csv" Q_CSV="${Q}_v2/quality_by_video.csv"
             ROUTE_ROOT="${ROUTE}" PROMPT_FILES="${PROMPTS}")
  env "${env[@]}" bash "${S}/run_univ_route.sh" table
  [[ -f "${ROUTE}/emb_t5.npz" ]] || env "${env[@]}" GPU="${GPUS[0]}" bash "${S}/run_univ_route.sh" embed-t5
  [[ -f "${ROUTE}/emb_qwen.npz" ]] || env "${env[@]}" GPU="${GPUS[1]}" bash "${S}/run_univ_route.sh" embed-qwen
  env "${env[@]}" bash "${S}/run_univ_route.sh" embed-eval
}

status() {
  echo "prompts:   $(wc -l < "${PROMPTS}" 2>/dev/null || echo 0)"
  echo "checklist: $(ls "${GEN}/raw" 2>/dev/null | wc -l) generated, $(wc -l < "${CHECKLIST}" 2>/dev/null || echo 0) valid"
  [[ -f "${P}/plan.json" ]] && PILOT_CONFIG="${CFG}/published_wan21_platform_natural.json" PUBLISHED_WAN21_ROOT="${P}" \
    bash "${S}/run_univ_published_wan21_8gpu.sh" status 2>&1 | tail -3 | sed 's/^/platform  /'
  [[ -f "${T}/plan.json" ]] && TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_natural.json" TWOSTAGE_ROOT="${T}" \
    REFERENCE_ROOT="${P}" bash "${S}/run_univ_twostage_wan21_8gpu.sh" status 2>&1 | sed 's/^/twostage  /'
  echo "fixed:     $(ls "${FIXED}/cache" 2>/dev/null | wc -l) clips"
  echo "dynamic:   vlm $(ls "${DYN}/vlm" 2>/dev/null | wc -l), pair $(ls "${DYN}/pair" 2>/dev/null | wc -l)"
  echo "quality:   vlm $(ls "${Q}/vlm" 2>/dev/null | wc -l), pair $(ls "${Q}/pair" 2>/dev/null | wc -l)"
  ls "${ROUTE}"_embed_analysis.tgz 2>/dev/null || true
  return 0
}

unattended() {
  # Each stage in its own bash so set -e holds inside it; a failure stops the chain (every later stage needs it).
  local stage
  for stage in checklist platform twostage fixed dynamic quality route; do
    bash "$0" "${stage}" || { say "STAGE FAILED: ${stage}"; status; exit 1; }
  done
  status
  local bundle="${OUT}/natural_v1_results.tgz" files=()
  mapfile -t files < <(cd "${OUT}" && ls -1 published_wan21_platform_natural/exports/*.tgz published_wan21_twostage_natural/exports/*.tgz \
    requirements_natural/checklist.jsonl requirements_natural/collect.txt fixed_metrics_natural_analysis.tgz \
    dynamic_natural_analysis.tgz quality_natural_analysis.tgz quality_natural_v2/quality_by_video.csv \
    quality_natural_v2/report.txt route_natural_embed_analysis.tgz natural_v1.log 2>/dev/null || true)
  tar czf "${bundle}" -C "${OUT}" "${files[@]}" && echo "SEND THIS: ${bundle}"
  say "NATURAL BATCH DONE"
}

case "${STAGE}" in
  all) unattended ;;
  prompts|checklist|platform|twostage|fixed|dynamic|quality|route|status) "${STAGE}" ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
