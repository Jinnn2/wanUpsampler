#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all-tfidf}"
case "${MODE}" in
  check|extract-proxy|embed|train-tfidf|train-t5|all-tfidf|all-t5) ;;
  *)
    echo "Usage: $0 [check|extract-proxy|embed|train-tfidf|train-t5|all-tfidf|all-t5]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-/opt/conda/bin/python}"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/router/train_sparse_b4_action_router.py"
BASE_DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/router/train_sparse_prompt_state_router.py"
SCORED_DIR="${SCORED_DIR:-${PROJECT_ROOT}/outputs/univ_sparse_action_phase3_v1/metrics/sparse_action_vbench}"
STATE_DIR="${STATE_DIR:-${SCORED_DIR}/reference_video_proxy}"
T5_DIR="${T5_DIR:-${SCORED_DIR}/t5_sparse_prompt_state}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
UTILITY_LAMBDA="${UTILITY_LAMBDA:-0.05}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.02}"
OBSERVATION_COST_RATIO="${OBSERVATION_COST_RATIO:-0.0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-250}"
FFMPEG="${FFMPEG:-ffmpeg}"
PROXY_SIZE="${PROXY_SIZE:-64}"
PROXY_FPS="${PROXY_FPS:-4}"
PROXY_FRAMES="${PROXY_FRAMES:-16}"
VERIFY_VIDEO_HASHES="${VERIFY_VIDEO_HASHES:-1}"
DEVICE="${DEVICE:-cuda}"

require_common() {
  [[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 1; }
  [[ -f "${DRIVER}" ]] || { echo "Driver not found: ${DRIVER}" >&2; exit 1; }
  [[ -f "${BASE_DRIVER}" ]] || { echo "Base driver not found: ${BASE_DRIVER}" >&2; exit 1; }
  for path in scored_dataset.json evaluation_inputs.json relative_quality_pairs.csv quality_by_video.csv; do
    [[ -f "${SCORED_DIR}/${path}" ]] || { echo "Scored input not found: ${SCORED_DIR}/${path}" >&2; exit 1; }
  done
}

check_inputs() {
  require_common
  "${PYTHON}" "${DRIVER}" check \
    --scored-dir "${SCORED_DIR}" \
    --utility-lambda "${UTILITY_LAMBDA}"
}

extract_proxy() {
  require_common
  command -v "${FFMPEG}" >/dev/null 2>&1 || { echo "ffmpeg not found: ${FFMPEG}" >&2; exit 1; }
  local -a args=(
    "${BASE_DRIVER}" extract-proxy
    --scored-dir "${SCORED_DIR}"
    --state-dir "${STATE_DIR}"
    --ffmpeg "${FFMPEG}"
    --proxy-size "${PROXY_SIZE}"
    --proxy-fps "${PROXY_FPS}"
    --proxy-frames "${PROXY_FRAMES}"
  )
  [[ "${VERIFY_VIDEO_HASHES}" == "1" ]] || args+=(--skip-video-hash)
  "${PYTHON}" "${args[@]}"
}

embed_prompts() {
  require_common
  [[ -d "${MODEL_ROOT}" ]] || { echo "Wan model root not found: ${MODEL_ROOT}" >&2; exit 1; }
  [[ -d "${LIGHTX2V_REPO}/lightx2v" ]] || { echo "LightX2V package not found: ${LIGHTX2V_REPO}/lightx2v" >&2; exit 1; }
  PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${BASE_DRIVER}" embed \
    --scored-dir "${SCORED_DIR}" \
    --t5-dir "${T5_DIR}" \
    --model-root "${MODEL_ROOT}" \
    --lightx2v-repo "${LIGHTX2V_REPO}" \
    --device "${DEVICE}"
}

train_router() {
  local prompt_features="$1"
  require_common
  [[ -f "${STATE_DIR}/state_manifest.json" ]] || { echo "State manifest not found: ${STATE_DIR}/state_manifest.json" >&2; exit 1; }
  local out_dir="${OUT_DIR:-${SCORED_DIR}/sparse_b4_${prompt_features}_lambda_${UTILITY_LAMBDA//./p}}"
  local -a args=(
    "${DRIVER}" train
    --scored-dir "${SCORED_DIR}"
    --state-dir "${STATE_DIR}"
    --out-dir "${out_dir}"
    --prompt-features "${prompt_features}"
    --utility-lambda "${UTILITY_LAMBDA}"
    --temperature "${SOFT_TEMPERATURE}"
    --observation-cost-ratio "${OBSERVATION_COST_RATIO}"
    --max-iterations "${MAX_ITERATIONS}"
  )
  if [[ "${prompt_features}" == "t5" ]]; then
    [[ -f "${T5_DIR}/t5_manifest.json" ]] || { echo "T5 manifest not found: ${T5_DIR}/t5_manifest.json" >&2; exit 1; }
    args+=(--t5-dir "${T5_DIR}")
  fi
  "${PYTHON}" "${args[@]}"
}

case "${MODE}" in
  check) check_inputs ;;
  extract-proxy) extract_proxy ;;
  embed) embed_prompts ;;
  train-tfidf) train_router tfidf ;;
  train-t5) train_router t5 ;;
  all-tfidf)
    check_inputs
    extract_proxy
    train_router tfidf
    ;;
  all-t5)
    check_inputs
    extract_proxy
    embed_prompts
    train_router t5
    ;;
esac

echo "B4-style output root: ${OUT_DIR:-${SCORED_DIR}/sparse_b4_<prompt_features>_lambda_${UTILITY_LAMBDA//./p}}"
