#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
USER_TAIL_SKIP_LMDB_DIR="${TAIL_SKIP_LORA_LMDB_DIR+x}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
START_OFFSET="${START_OFFSET:-0}"
TRAIN_STEP="${TRAIN_STEP:-45}"
INFER_STEPS="${INFER_STEPS:-50}"
MODE="${MODE:-lightx2v}"
PRECISION="${PRECISION:-bf16}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-8}"

SOURCE_LMDB="${CR_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
if [[ -n "${USER_TAIL_SKIP_LMDB_DIR}" ]]; then
  LMDB_ROOT="${TAIL_SKIP_LORA_LMDB_DIR}"
else
  LMDB_ROOT="${PROJECT_ROOT}/data/changing_resolution/lmdb_tail_skip_lora_step${TRAIN_STEP}_to_step${INFER_STEPS}"
fi
PARTS_DIR="${LMDB_ROOT}/_parts"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_tail_skip_lora_lmdb_step${TRAIN_STEP}_multigpu}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
if (( NUM_GPUS < 1 )); then
  echo "GPU_IDS is empty" >&2
  exit 2
fi
if (( TOTAL_SAMPLES < 1 )); then
  echo "TOTAL_SAMPLES must be positive, got ${TOTAL_SAMPLES}" >&2
  exit 2
fi
if (( START_OFFSET < 0 )); then
  echo "START_OFFSET must be non-negative, got ${START_OFFSET}" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_LMDB}" ]] || [[ -z "$(find "${SOURCE_LMDB}" -type f -name 'data.mdb' -print -quit 2>/dev/null)" ]]; then
  echo "No source clean LMDB shards found under: ${SOURCE_LMDB}" >&2
  exit 1
fi

count_lmdb_samples() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi
  python - "${root}" <<'PY'
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
}

count_part_samples() {
  local part_name="$1"
  local total=0
  local candidate
  for candidate in "${PARTS_DIR}/${part_name}" "${PARTS_DIR}/${part_name}"_resume_*; do
    [[ -d "${candidate}" ]] || continue
    total=$((total + $(count_lmdb_samples "${candidate}")))
  done
  echo "${total}"
}

mkdir -p "${LMDB_ROOT}" "${PARTS_DIR}" "${LOG_DIR}"

if [[ "${OVERWRITE}" == "1" ]]; then
  find "${LMDB_ROOT}" -mindepth 1 -maxdepth 1 \( -name 'shard_*' -o -name '_parts' \) -exec rm -rf {} +
  mkdir -p "${PARTS_DIR}"
fi

echo "Multi-GPU tail-skip LoRA LMDB build"
echo "  project      : ${PROJECT_ROOT}"
echo "  source_lmdb  : ${SOURCE_LMDB}"
echo "  lmdb_root    : ${LMDB_ROOT}"
echo "  total_samples: ${TOTAL_SAMPLES}"
echo "  start_offset : ${START_OFFSET}"
echo "  train_step   : ${TRAIN_STEP}"
echo "  infer_steps  : ${INFER_STEPS}"
echo "  mode         : ${MODE}"
echo "  precision    : ${PRECISION}"
echo "  resume       : ${RESUME}"
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
  existing_count="$(count_part_samples "${part_name}")"
  launch_offset="${offset}"
  launch_count="${count}"

  if [[ "${RESUME}" == "1" ]]; then
    if (( existing_count == count )); then
      echo "Skip ${part_name}: already complete (${existing_count}/${count} samples)"
      offset=$((offset + count))
      continue
    fi
    if (( existing_count > count )); then
      echo "Existing data for ${part_name} exceeds expected samples: ${existing_count}/${count}" >&2
      echo "Use OVERWRITE=1 if you want to rebuild from scratch." >&2
      exit 1
    fi
    if (( existing_count > 0 )); then
      launch_offset=$((offset + existing_count))
      launch_count=$((count - existing_count))
      part_lmdb="${PARTS_DIR}/${part_name}_resume_${existing_count}"
      log_path="${LOG_DIR}/${part_name}_resume_${existing_count}.log"
      echo "Resume ${part_name}: existing=${existing_count}/${count}, offset=${launch_offset}, remaining=${launch_count}"
    fi
  elif (( existing_count > 0 )); then
    echo "Existing partial data found for ${part_name}: ${part_lmdb} (${existing_count}/${count} samples)" >&2
    echo "Use RESUME=1 to keep existing samples and build only the missing tail, or OVERWRITE=1 to rebuild everything." >&2
    exit 1
  fi

  echo "Launch ${part_name}: gpu=${gpu}, offset=${launch_offset}, count=${launch_count}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CR_STAGE2_LMDB_DIR="${SOURCE_LMDB}" \
    TAIL_SKIP_LORA_LMDB_DIR="${part_lmdb}" \
    TRAIN_STEP="${TRAIN_STEP}" \
    INFER_STEPS="${INFER_STEPS}" \
    SAMPLE_OFFSET="${launch_offset}" \
    MAX_SAMPLES="${launch_count}" \
    MODE="${MODE}" \
    PRECISION="${PRECISION}" \
    OVERWRITE="${OVERWRITE}" \
    bash changing_resolution/scripts/data/build_tail_skip_lora_lmdb.sh
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

if (( remaining == 0 )); then
  echo "No workers launched; all part LMDBs were already complete."
fi

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
      if [[ "${RESUME}" == "1" && -d "${dst}" && -f "${dst}/data.mdb" ]]; then
        echo "Skip already merged shard: ${dst}"
        continue
      fi
      echo "Merged shard already exists: ${dst}" >&2
      echo "Set RESUME=1 to skip already merged shards, or OVERWRITE=1 to rebuild and merge from scratch." >&2
      exit 1
    fi
    mv "${shard_dir}" "${dst}"
  done
done

shard_count="$(find "${LMDB_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | wc -l)"
sample_count="$(count_lmdb_samples "${LMDB_ROOT}")"

echo "Merged tail-skip LoRA LMDB ready: ${LMDB_ROOT}"
echo "  shards : ${shard_count}"
echo "  samples: ${sample_count}"

if (( sample_count != TOTAL_SAMPLES )); then
  echo "Expected ${TOTAL_SAMPLES} samples, got ${sample_count}" >&2
  exit 1
fi
