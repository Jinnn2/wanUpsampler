#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VIDEO_DIR="${VIDEO_DIR:?Set VIDEO_DIR to generated HR videos}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean}"
PARTS_DIR="${OUT_DIR}/_parts"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-0}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
WAN_REPO="${WAN_REPO:-${LIGHTX2V_REPO:-}}"
HR_H="${HR_H:-720}"; HR_W="${HR_W:-1248}"
SCALES="${SCALES:-1.5 2.0 3.0}"
LR_SIZES="${LR_SIZES:-}"
NUM_FRAMES="${NUM_FRAMES:-81}"
PRECISION="${PRECISION:-bf16}"
VAE_BACKEND="${VAE_BACKEND:-auto}"
RESIZE_MODE="${RESIZE_MODE:-bicubic}"
MAP_SIZE_GB="${MAP_SIZE_GB:-128}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_uni_build}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
if (( TOTAL_SAMPLES < 1 )); then
  echo "TOTAL_SAMPLES must be set to a positive number for deterministic sharding." >&2
  exit 2
fi
mkdir -p "${PARTS_DIR}" "${LOG_DIR}"
if [[ -n "$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' -print -quit 2>/dev/null)" || -n "$(find "${PARTS_DIR}" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
  echo "Output already contains parts. Remove or choose a new OUT_DIR." >&2
  exit 2
fi

base=$((TOTAL_SAMPLES / NUM_GPUS)); rem=$((TOTAL_SAMPLES % NUM_GPUS))
pids=(); offset=0
for ((rank=0; rank<NUM_GPUS; rank++)); do
  count="${base}"; if (( rank < rem )); then count=$((count + 1)); fi
  part="${PARTS_DIR}/part_$(printf '%02d' "${rank}")"
  log="${LOG_DIR}/part_$(printf '%02d' "${rank}").log"
  args=(
    -m changing_resolution_uni.build_latent_pairs
    --video_dir "${VIDEO_DIR}" --out_dir "${part}"
    --model_root "${MODEL_ROOT}" --vae_path "${VAE_PATH}"
    --hr_size "${HR_H}" "${HR_W}" --scales ${SCALES}
    --num_frames "${NUM_FRAMES}" --video_offset "${offset}"
    --device cuda --precision "${PRECISION}" --map_size_gb "${MAP_SIZE_GB}"
    --vae_backend "${VAE_BACKEND}" --resize_mode "${RESIZE_MODE}"
  )
  if [[ -n "${LR_SIZES}" ]]; then args+=(--lr_sizes ${LR_SIZES}); fi
  if [[ -n "${WAN_REPO}" ]]; then args+=(--wan_repo "${WAN_REPO}"); fi
  if (( count > 0 )); then args+=(--max_samples "${count}"); fi
  echo "launch rank=${rank} gpu=${GPUS[$rank]} offset=${offset} count=${count}"
  (cd "${PROJECT_ROOT}" && CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" python "${args[@]}") >"${log}" 2>&1 &
  pids+=("$!")
  if (( count > 0 )); then offset=$((offset + count)); fi
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if (( failed != 0 )); then
  echo "A data worker failed; inspect ${LOG_DIR}" >&2
  exit 1
fi

for part in "${PARTS_DIR}"/part_*; do
  [[ -d "${part}/shard_00000" ]] || continue
  name="$(basename "${part}")"
  mv "${part}/shard_00000" "${OUT_DIR}/shard_${name#part_}"
  rmdir "${part}"
done
rmdir "${PARTS_DIR}" 2>/dev/null || true
echo "Universal clean LMDB shards ready under ${OUT_DIR}"
