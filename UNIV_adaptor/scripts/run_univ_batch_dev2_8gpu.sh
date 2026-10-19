#!/usr/bin/env bash
# One ~10 h batch on one 8-GPU machine (see UNIV_adaptor/PUBLISHED_WAN21_BATCH_DEV2.md):
#   v2b       rCM 4/2/1 steps + LR41->FlashVSR->RIFE on the v1 dev prompts (FULL50 reused from v2a)
#   platform  the 6 platform-v1 arms on the 64 NEW dev2 prompts (frozen v1 driver, new config)
#   twostage  all v2a + v2b arms on dev2 (FULL50 reused from platform)
#   dvg       DVG 0.7 / 0.5 (+ its own FULL) on dev2
#   requirements  RAFT flow / detail / VLM checklist for v2b, platform, twostage, dvg
#
#   bash UNIV_adaptor/scripts/run_univ_batch_dev2_8gpu.sh all        # everything, in this order
#   bash UNIV_adaptor/scripts/run_univ_batch_dev2_8gpu.sh <stage>    # one stage; every stage resumes from receipts
#   bash UNIV_adaptor/scripts/run_univ_batch_dev2_8gpu.sh status
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-help}"
OUT="${REPO_ROOT}/outputs"
S="${REPO_ROOT}/UNIV_adaptor/scripts"
CFG="${REPO_ROOT}/UNIV_adaptor/configs"
V2A="${OUT}/published_wan21_twostage_v1"
V2B="${OUT}/published_wan21_twostage_v2b"
P2="${OUT}/published_wan21_platform_dev2"
T2="${OUT}/published_wan21_twostage_dev2"
D2="${OUT}/dvg_wan21_study_dev2"
CHECK_V1="${REPO_ROOT}/prompts/univ_controlled_factor_v1_requirements.jsonl"
CHECK_DEV2="${REPO_ROOT}/prompts/univ_controlled_factor_dev2_requirements.jsonl"
ALL_GPUS="0,1,2,3,4,5,6,7"
export NGPUS=8
# Interpreters verified by install_twostage_wan21_env.sh on this machine also drive the v1 and DVG wrappers.
GPU_TAG="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
ENV_FILE="${TWOSTAGE_ENV_FILE:-${OUT}/twostage_env/${GPU_TAG}.env}"
[[ -f "${ENV_FILE}" ]] || { echo "Missing ${ENV_FILE}; run install_twostage_wan21_env.sh first" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_FILE}"
export TWOSTAGE_ENV_FILE="${ENV_FILE}" PYTHON_BIN="${WAN_PYTHON}" GEN_PYTHON="${WAN_PYTHON}"
export VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
cd "${REPO_ROOT}"
say() { printf '\n######## %s  (%s)\n' "$*" "$(date '+%F %T')"; }

v2b() {
  say "v2b: rCM + LR41->FlashVSR->RIFE on v1 dev prompts"
  bash "${S}/install_twostage_wan21_env.sh" rcm
  [[ -f "${V2A}/dataset_manifest.json" ]] || { echo "v2a must be finalized first: ${V2A}" >&2; exit 1; }
  TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_v2b.json" TWOSTAGE_ROOT="${V2B}" REFERENCE_ROOT="${V2A}" \
    bash "${S}/run_univ_twostage_wan21_8gpu.sh" all
}

platform() {
  say "platform dev2: STEP25/STEP20/TEA008/TEA015/SCALING10 + FULL50 on dev2 prompts"
  local m
  for m in check calibrate generate finalize score export; do
    say "platform ${m}"
    PILOT_CONFIG="${CFG}/published_wan21_platform_dev2.json" PUBLISHED_WAN21_ROOT="${P2}" \
      bash "${S}/run_univ_published_wan21_8gpu.sh" "${m}"
  done
}

twostage() {
  say "two-stage dev2: v2a + v2b arms on dev2 prompts"
  bash "${S}/install_twostage_wan21_env.sh" rcm
  [[ -f "${P2}/dataset_manifest.json" ]] || { echo "platform dev2 must be finalized first: ${P2}" >&2; exit 1; }
  TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_dev2.json" TWOSTAGE_ROOT="${T2}" REFERENCE_ROOT="${P2}" \
    bash "${S}/run_univ_twostage_wan21_8gpu.sh" all
}

dvg() {
  say "DVG dev2"
  local m
  for m in plan generate score export; do
    say "dvg ${m}"
    [[ "${m}" == plan && -f "${D2}/plan.json" ]] && continue
    STUDY_ROOT="${D2}" PROMPTS="${REPO_ROOT}/prompts/univ_controlled_factor_dev2.jsonl" SETS=main GPU_IDS="${ALL_GPUS}" \
      bash "${S}/run_univ_dvg_wan21_study.sh" "${m}"
  done
}

requirements() {
  say "requirements: flow / detail / VLM checklist"
  local root req checklist failed=0
  for spec in "${V2A}:platform_v2a_requirements:${CHECK_V1}" "${V2B}:platform_v2b_requirements:${CHECK_V1}" \
              "${P2}:platform_dev2_requirements:${CHECK_DEV2}" "${T2}:twostage_dev2_requirements:${CHECK_DEV2}"; do
    IFS=: read -r root req checklist <<< "${spec}"
    [[ -f "${root}/metrics/quality_by_video.csv" ]] || { echo "skip (not scored yet): ${root}"; continue; }
    say "requirements ${req}"
    # One root failing (e.g. RAFT download) must not stop the others.
    if ! (export PUBLISHED_ROOT="${root}" REQ_ROOT="${OUT}/${req}" CHECKLIST="${checklist}" GPU_IDS="${ALL_GPUS}"
          R="${S}/run_univ_controlled_factor_requirements.sh"
          bash "$R" plan && bash "$R" signals && bash "$R" vlm && bash "$R" report); then
      echo "REQUIREMENTS FAILED: ${req}" >&2
      failed=1
    fi
  done
  if [[ -f "${D2}/quality_by_video.csv" ]]; then
    say "requirements dvg_dev2"
    STUDY_ROOT="${D2}" REQ_ROOT="${OUT}/dvg_dev2_requirements" CHECKLIST="${CHECK_DEV2}" GPU_IDS="${ALL_GPUS}" \
      bash "${S}/run_univ_dvg_wan21_study.sh" requirements || { echo "REQUIREMENTS FAILED: dvg_dev2" >&2; failed=1; }
  fi
  return "${failed}"
}

unattended() {
  # Every stage in its own bash (so set -e holds inside it); a failed stage does not stop later
  # independent stages. twostage needs platform, and both check that themselves.
  local stage failed=()
  for stage in v2b platform twostage dvg requirements; do
    bash "$0" "${stage}" || { failed+=("${stage}"); say "STAGE FAILED: ${stage} (continuing)"; }
  done
  status
  # Everything to send back, in one file (analysis bundles only, no videos), plus the batch log.
  local bundle="${OUT}/batch_dev2_results.tgz" files=()
  mapfile -t files < <(cd "${OUT}" && ls -1 published_wan21_twostage_v2b/exports/*.tgz published_wan21_platform_dev2/exports/*.tgz \
    published_wan21_twostage_dev2/exports/*.tgz dvg_wan21_study_dev2_analysis.tgz *_requirements_analysis.tgz \
    batch_dev2.log 2>/dev/null || true)
  (( ${#files[@]} )) && tar czf "${bundle}" -C "${OUT}" "${files[@]}" && echo "SEND THIS: ${bundle}"
  say "BATCH DONE. failed stages: ${failed[*]:-none}"
}

status() {
  [[ -f "${V2B}/plan.json" ]] && TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_v2b.json" TWOSTAGE_ROOT="${V2B}" \
    REFERENCE_ROOT="${V2A}" bash "${S}/run_univ_twostage_wan21_8gpu.sh" status | sed 's/^/v2b      /'
  [[ -f "${P2}/plan.json" ]] && PILOT_CONFIG="${CFG}/published_wan21_platform_dev2.json" PUBLISHED_WAN21_ROOT="${P2}" \
    bash "${S}/run_univ_published_wan21_8gpu.sh" status 2>&1 | tail -3 | sed 's/^/platform /'
  [[ -f "${T2}/plan.json" ]] && TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_dev2.json" TWOSTAGE_ROOT="${T2}" \
    REFERENCE_ROOT="${P2}" bash "${S}/run_univ_twostage_wan21_8gpu.sh" status | sed 's/^/twostage /'
  [[ -f "${D2}/plan.json" ]] && STUDY_ROOT="${D2}" bash "${S}/run_univ_dvg_wan21_study.sh" status 2>&1 | tail -2 | sed 's/^/dvg      /'
  for f in "${V2B}" "${P2}" "${T2}"; do ls "${f}"/exports/*.tgz 2>/dev/null || true; done
  ls "${OUT}"/*dev2*_analysis.tgz "${OUT}"/*v2b*_analysis.tgz 2>/dev/null || true
  return 0
}

case "${STAGE}" in
  all) unattended ;;
  v2b|platform|twostage|dvg|requirements|status) "${STAGE}" ;;
  *) echo "Stages: all v2b platform twostage dvg requirements status"; exit 2 ;;
esac
