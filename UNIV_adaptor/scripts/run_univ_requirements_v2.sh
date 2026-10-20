#!/usr/bin/env bash
# M1 of REQUIREMENTS_V2_SPEC.md: dynamic-layer checklists written by a large VLM (text only). No video work.
#   bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh setup    # venv + model download (once)
#   bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh check    # one prompt on one GPU
#   bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh pilot    # 12 held-out pilot prompts, recall vs reviewed
#   bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh full     # every dev + dev2 prompt (128)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
GEN_PYTHON="${GEN_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
GEN_MODEL_ID="${GEN_MODEL_ID:-Qwen/Qwen3-VL-32B-Instruct}"
GEN_MODEL="${GEN_MODEL:-/mnt/afs_2/houze/models/$(basename "${GEN_MODEL_ID}")}"
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/outputs/requirements_v2_gen}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-1}"  # 32B fits one H100; use 2 for a 72B model
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/requirements_v2.py"
PROMPTS=("${REPO_ROOT}/prompts/univ_controlled_factor_v1.jsonl" "${REPO_ROOT}/prompts/univ_controlled_factor_dev2.jsonl")
PILOT="${REPO_ROOT}/prompts/univ_requirements_v2_pilot.jsonl"
# Few-shot examples are the watchmaking family of the pilot; the other 12 pilot prompts measure recall.
HELD_OUT=(1 11 12 13 100 101 102 103 104 105 106 107)
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"

fan_out() {  # $1 = output dir, rest = extra generate args; one worker per GPUS_PER_WORKER GPUs
  local out="$1" pids=() i n
  shift
  n=$(( ${#GPUS[@]} / GPUS_PER_WORKER ))
  mkdir -p "${out}/logs"
  for (( i = 0; i < n; i++ )); do
    local devs
    devs="$(IFS=,; echo "${GPUS[*]:$(( i * GPUS_PER_WORKER )):${GPUS_PER_WORKER}}")"
    CUDA_VISIBLE_DEVICES="${devs}" "${GEN_PYTHON}" "${DRIVER}" generate --prompts "${PROMPTS[@]}" \
      --examples "${PILOT}" --model "${GEN_MODEL}" --out "${out}" --shard-index "${i}" --shard-count "${n}" "$@" \
      > "${out}/logs/generate_${i}.log" 2>&1 &
    pids+=("$!")
  done
  trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
  local failed=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "worker ${i} failed: ${out}/logs/generate_${i}.log" >&2; failed=1; }
  done
  return "${failed}"
}

case "${MODE}" in
  setup)
    df -h "$(dirname "${GEN_MODEL}")"
    [[ -x "${GEN_PYTHON}" ]] || "${BASE_PYTHON}" -m venv --system-site-packages "${REPO_ROOT}/.venv_vlm3"
    "${GEN_PYTHON}" -m pip install "transformers>=4.57" "accelerate>=1.0"
    [[ -d "${GEN_MODEL}" ]] || "${GEN_PYTHON}" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])' \
      "${GEN_MODEL_ID}" "${GEN_MODEL}"
    ;;
  check)
    rm -rf "${GEN_ROOT}/check"
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]:0:${GPUS_PER_WORKER}}")" "${GEN_PYTHON}" "${DRIVER}" generate \
      --prompts "${PROMPTS[@]}" --examples "${PILOT}" --model "${GEN_MODEL}" --out "${GEN_ROOT}/check" --only 100
    "${BASE_PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r["record"], indent=1)); [print("attempt", i, a["errors"]) for i, a in enumerate(r["attempts"])]' \
      "${GEN_ROOT}/check/raw/100.json"
    ;;
  pilot)
    rm -rf "${GEN_ROOT}/pilot"  # a measurement: always from scratch (generate skips prompts already in raw/)
    fan_out "${GEN_ROOT}/pilot" --only "${HELD_OUT[@]}"
    "${BASE_PYTHON}" "${DRIVER}" collect --prompts "${PROMPTS[@]}" --out "${GEN_ROOT}/pilot" | tee "${GEN_ROOT}/pilot/collect.txt"
    "${BASE_PYTHON}" "${DRIVER}" compare --generated "${GEN_ROOT}/pilot/checklist.jsonl" --reference "${PILOT}" \
      --prompts "${PROMPTS[@]}" --report "${GEN_ROOT}/pilot/compare.md" > /dev/null
    sed -n 1,25p "${GEN_ROOT}/pilot/compare.md"
    tar czf "${GEN_ROOT}_pilot.tgz" -C "$(dirname "${GEN_ROOT}")" "$(basename "${GEN_ROOT}")/pilot"
    echo "Download: ${GEN_ROOT}_pilot.tgz"
    ;;
  full)
    fan_out "${GEN_ROOT}/full"
    "${BASE_PYTHON}" "${DRIVER}" collect --prompts "${PROMPTS[@]}" --out "${GEN_ROOT}/full" | tee "${GEN_ROOT}/full/collect.txt"
    tar czf "${GEN_ROOT}_full.tgz" -C "$(dirname "${GEN_ROOT}")" "$(basename "${GEN_ROOT}")/full"
    echo "Download: ${GEN_ROOT}_full.tgz"
    ;;
  *)
    echo "Usage: bash $0 {setup|check|pilot|full}"
    echo "Model: ${GEN_MODEL_ID} -> ${GEN_MODEL}; workers: ${#GPUS[@]} GPUs / ${GPUS_PER_WORKER} per worker"
    ;;
esac
