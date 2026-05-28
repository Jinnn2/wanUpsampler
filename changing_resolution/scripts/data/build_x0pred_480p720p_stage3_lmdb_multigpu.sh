#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
START_OFFSET="${START_OFFSET:-0}"
DENOISE_STEP="${DENOISE_STEP:-45}"
HR_TARGET_MODE="${HR_TARGET_MODE:-x0_pred}"
HR_SEED_OFFSET="${HR_SEED_OFFSET:-0}"

SOURCE_LMDB="${CR_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
LMDB_ROOT="${CR_STAGE3_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_x0pred_480p720p_stage3_step${DENOISE_STEP}}"
PARTS_DIR="${LMDB_ROOT}/_parts"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_stage3_x0pred_lmdb_step${DENOISE_STEP}_multigpu}"
OVERWRITE="${OVERWRITE:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-8}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
if (( NUM_GPUS < 1 )); then
  echo "GPU_IDS is empty" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_LMDB}" ]] || [[ -z "$(find "${SOURCE_LMDB}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
  echo "No source clean LMDB shards found under: ${SOURCE_LMDB}" >&2
  exit 1
fi

mkdir -p "${LMDB_ROOT}" "${PARTS_DIR}" "${LOG_DIR}"

if [[ "${OVERWRITE}" == "1" ]]; then
  find "${LMDB_ROOT}" -mindepth 1 -maxdepth 1 \( -name 'shard_*' -o -name '_parts' \) -exec rm -rf {} +
  mkdir -p "${PARTS_DIR}"
fi

echo "Multi-GPU Stage 3 x0-pred LMDB build"
echo "  project      : ${PROJECT_ROOT}"
echo "  source_lmdb  : ${SOURCE_LMDB}"
echo "  lmdb_root    : ${LMDB_ROOT}"
echo "  total_samples: ${TOTAL_SAMPLES}"
echo "  start_offset : ${START_OFFSET}"
echo "  denoise_step : ${DENOISE_STEP}"
echo "  hr_target    : ${HR_TARGET_MODE}"
echo "  gpu_ids      : ${GPU_IDS}"
echo "  log_dir      : ${LOG_DIR}"

base_count=$((TOTAL_SAMPLES / NUM_GPUS))
remainder=$((TOTAL_SAMPLES % NUM_GPUS))
offset="${START_OFFSET}"
pids=()
worker_names=()
worker_logs=()

cleanup_workers() {
  if (( ${#pids[@]} > 0 )); then
    echo "Stopping GPU workers..." >&2
    kill "${pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_workers; exit 130' INT TERM

for rank in "${!GPUS[@]}"; do
  count="${base_count}"
  if (( rank < remainder )); then
    count=$((count + 1))
  fi
  if (( count == 0 )); then
    continue
  fi

  gpu="${GPUS[$rank]}"
  part_name="$(printf "part_%02d" "${rank}")"
  part_lmdb="${PARTS_DIR}/${part_name}"
  log_path="${LOG_DIR}/${part_name}.log"

  echo "Launch ${part_name}: gpu=${gpu}, offset=${offset}, count=${count}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CR_STAGE2_LMDB_DIR="${SOURCE_LMDB}" \
    CR_STAGE3_LMDB_DIR="${part_lmdb}" \
    DENOISE_STEP="${DENOISE_STEP}" \
    HR_TARGET_MODE="${HR_TARGET_MODE}" \
    HR_SEED_OFFSET="${HR_SEED_OFFSET}" \
    SAMPLE_OFFSET="${offset}" \
    MAX_SAMPLES="${count}" \
    OVERWRITE="${OVERWRITE}" \
    bash changing_resolution/scripts/data/build_x0pred_480p720p_stage3_lmdb.sh
  ) >"${log_path}" 2>&1 &

  pids+=("$!")
  worker_names+=("${part_name}")
  worker_logs+=("${log_path}")
  offset=$((offset + count))
done

failed=0
remaining="${#pids[@]}"
finished=()
for _ in "${pids[@]}"; do
  finished+=(0)
done

echo "Workers launched. Logs are under: ${LOG_DIR}"
echo "The launcher prints the tail of each active worker log every ${MONITOR_INTERVAL}s."

while (( remaining > 0 )); do
  sleep "${MONITOR_INTERVAL}"
  echo "----- worker status $(date '+%F %T') -----"
  for i in "${!pids[@]}"; do
    if (( finished[i] == 1 )); then
      continue
    fi
    pid="${pids[$i]}"
    name="${worker_names[$i]}"
    log_path="${worker_logs[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[running] ${name} pid=${pid} log=${log_path}"
      if [[ -f "${log_path}" ]]; then
        tail -n "${MONITOR_TAIL_LINES}" "${log_path}" || true
      else
        echo "(log not created yet)"
      fi
    else
      if wait "${pid}"; then
        echo "[done] ${name}"
      else
        echo "[failed] ${name}. Last log lines:"
        if [[ -f "${log_path}" ]]; then
          tail -n 80 "${log_path}" || true
        fi
        failed=1
      fi
      finished[i]=1
      remaining=$((remaining - 1))
    fi
  done
done

trap - INT TERM

if (( failed != 0 )); then
  echo "At least one GPU worker failed. Check logs under: ${LOG_DIR}" >&2
  exit 1
fi

echo "Merging part shards into ${LMDB_ROOT}"
for part_dir in "${PARTS_DIR}"/part_*; do
  [[ -d "${part_dir}" ]] || continue
  part_base="$(basename "${part_dir}")"
  for shard_dir in "${part_dir}"/shard_*; do
    [[ -d "${shard_dir}" ]] || continue
    shard_base="$(basename "${shard_dir}")"
    dst="${LMDB_ROOT}/shard_${part_base}_${shard_base#shard_}"
    if [[ -e "${dst}" ]]; then
      echo "Merged shard already exists: ${dst}" >&2
      echo "Set OVERWRITE=1 to rebuild and merge from scratch." >&2
      exit 1
    fi
    mv "${shard_dir}" "${dst}"
  done
done

shard_count="$(find "${LMDB_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | wc -l)"
sample_count="$(python - "${LMDB_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import lmdb

root = Path(sys.argv[1])
total = 0
for shard in sorted(root.glob("shard_*")):
    if not (shard / "data.mdb").exists():
        continue
    env = lmdb.open(str(shard), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            raw = txn.get(b"metadata")
            if raw:
                total += int(json.loads(raw.decode("utf-8"))["num_samples"])
    finally:
        env.close()
print(total)
PY
)"

echo "Merged Stage 3 LMDB ready: ${LMDB_ROOT}"
echo "  shards : ${shard_count}"
echo "  samples: ${sample_count}"

if (( sample_count != TOTAL_SAMPLES )); then
  echo "Expected ${TOTAL_SAMPLES} samples, got ${sample_count}" >&2
  exit 1
fi
