#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
USER_CR_DISTILL_LORA_LMDB_DIR="${CR_DISTILL_LORA_LMDB_DIR+x}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
START_OFFSET="${START_OFFSET:-0}"
OVERWRITE="${OVERWRITE:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-8}"

if [[ -n "${USER_CR_DISTILL_LORA_LMDB_DIR}" ]]; then
  LMDB_ROOT="${CR_DISTILL_LORA_LMDB_DIR}"
else
  LMDB_ROOT="${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3"
fi
PARTS_DIR="${LMDB_ROOT}/_parts"
RESUME_DIR="${LMDB_ROOT}/_resume"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_distill_last_step_skip_lora_resume}"

if [[ "${OVERWRITE}" == "1" ]]; then
  echo "Refusing to resume with OVERWRITE=1 because it can delete existing work." >&2
  echo "Run the original multigpu script with OVERWRITE=1 only if you want a full rebuild." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
if (( NUM_GPUS < 1 )); then
  echo "GPU_IDS is empty" >&2
  exit 2
fi

mkdir -p "${LMDB_ROOT}" "${PARTS_DIR}" "${RESUME_DIR}" "${LOG_DIR}"

count_rank_samples() {
  local rank="$1"
  local part_name
  part_name="$(printf "part_%02d" "${rank}")"
  python - "${LMDB_ROOT}" "${PARTS_DIR}/${part_name}" "${RESUME_DIR}/${part_name}" "${part_name}" <<'PY'
import json
import sys
from pathlib import Path

import lmdb

root = Path(sys.argv[1])
part_dir = Path(sys.argv[2])
resume_part_dir = Path(sys.argv[3])
part_name = sys.argv[4]


def shard_samples(shard: Path) -> int:
    if not (shard / "data.mdb").exists():
        return 0
    env = lmdb.open(str(shard), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            raw = txn.get(b"metadata")
            if not raw:
                return 0
            return int(json.loads(raw.decode("utf-8")).get("num_samples", 0))
    finally:
        env.close()


total = 0
for shard in sorted(root.glob(f"shard_{part_name}_*")):
    total += shard_samples(shard)
for shard in sorted(part_dir.glob("shard_*")):
    total += shard_samples(shard)
if resume_part_dir.exists():
    for run_dir in sorted(resume_part_dir.glob("run_*")):
        for shard in sorted(run_dir.glob("shard_*")):
            total += shard_samples(shard)
print(total)
PY
}

merge_shards() {
  local rank="$1"
  local part_name
  part_name="$(printf "part_%02d" "${rank}")"

  local shard_dir shard_base dst run_dir run_base
  if [[ -d "${PARTS_DIR}/${part_name}" ]]; then
    for shard_dir in "${PARTS_DIR}/${part_name}"/shard_*; do
      [[ -d "${shard_dir}" ]] || continue
      shard_base="$(basename "${shard_dir}")"
      dst="${LMDB_ROOT}/shard_${part_name}_${shard_base#shard_}"
      if [[ -e "${dst}" ]]; then
        echo "Merge target already exists: ${dst}" >&2
        echo "Leaving source in place: ${shard_dir}" >&2
        exit 1
      fi
      mv "${shard_dir}" "${dst}"
    done
  fi

  if [[ -d "${RESUME_DIR}/${part_name}" ]]; then
    for run_dir in "${RESUME_DIR}/${part_name}"/run_*; do
      [[ -d "${run_dir}" ]] || continue
      run_base="$(basename "${run_dir}")"
      for shard_dir in "${run_dir}"/shard_*; do
        [[ -d "${shard_dir}" ]] || continue
        shard_base="$(basename "${shard_dir}")"
        dst="${LMDB_ROOT}/shard_${part_name}_${run_base}_${shard_base#shard_}"
        if [[ -e "${dst}" ]]; then
          echo "Merge target already exists: ${dst}" >&2
          echo "Leaving source in place: ${shard_dir}" >&2
          exit 1
        fi
        mv "${shard_dir}" "${dst}"
      done
    done
  fi
}

sample_count() {
  python - "${LMDB_ROOT}" <<'PY'
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
                total += int(json.loads(raw.decode("utf-8")).get("num_samples", 0))
    finally:
        env.close()
print(total)
PY
}

echo "Resume multi-GPU last-step-skip LoRA LMDB build"
echo "  project      : ${PROJECT_ROOT}"
echo "  lmdb_root    : ${LMDB_ROOT}"
echo "  total_samples: ${TOTAL_SAMPLES}"
echo "  start_offset : ${START_OFFSET}"
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
    echo "Stopping resume workers..." >&2
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

  done_count="$(count_rank_samples "${rank}")"
  if (( done_count > count )); then
    part_name="$(printf "part_%02d" "${rank}")"
    echo "${part_name} has ${done_count} samples, more than assigned count ${count}." >&2
    exit 1
  fi

  remaining=$((count - done_count))
  part_name="$(printf "part_%02d" "${rank}")"
  echo "${part_name}: assigned_offset=${offset}, assigned_count=${count}, done=${done_count}, remaining=${remaining}"
  if (( remaining == 0 )); then
    offset=$((offset + count))
    continue
  fi

  gpu="$(echo "${GPUS[$rank]}" | xargs)"
  run_name="run_$(date '+%Y%m%d_%H%M%S')_$$"
  run_dir="${RESUME_DIR}/${part_name}/${run_name}"
  log_path="${LOG_DIR}/${part_name}_${run_name}.log"
  mkdir -p "${run_dir}"

  echo "Launch resume ${part_name}: gpu=${gpu}, offset=$((offset + done_count)), count=${remaining}, out=${run_dir}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CR_DISTILL_LORA_LMDB_DIR="${run_dir}" \
    SAMPLE_OFFSET="$((offset + done_count))" \
    MAX_SAMPLES="${remaining}" \
    OVERWRITE=1 \
    bash changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb.sh
  ) >"${log_path}" 2>&1 &

  pids+=("$!")
  worker_names+=("${part_name}")
  worker_logs+=("${log_path}")
  offset=$((offset + count))
done

failed=0
remaining_workers="${#pids[@]}"
finished=()
for _ in "${pids[@]}"; do
  finished+=(0)
done

if (( remaining_workers > 0 )); then
  echo "Resume workers launched. Logs are under: ${LOG_DIR}"
else
  echo "No resume workers needed; all assigned samples are already present."
fi

while (( remaining_workers > 0 )); do
  sleep "${MONITOR_INTERVAL}"
  echo "----- resume worker status $(date '+%F %T') -----"
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
      remaining_workers=$((remaining_workers - 1))
    fi
  done
done

trap - INT TERM

if (( failed != 0 )); then
  echo "At least one resume worker failed. Re-run this script after fixing the error." >&2
  exit 1
fi

offset="${START_OFFSET}"
for rank in "${!GPUS[@]}"; do
  count="${base_count}"
  if (( rank < remainder )); then
    count=$((count + 1))
  fi
  [[ "${count}" == "0" ]] && continue
  done_count="$(count_rank_samples "${rank}")"
  part_name="$(printf "part_%02d" "${rank}")"
  echo "${part_name}: after resume done=${done_count}/${count}"
  if (( done_count != count )); then
    echo "${part_name} is still incomplete; re-run this script to continue." >&2
    exit 1
  fi
  offset=$((offset + count))
done

echo "Merging available part and resume shards into ${LMDB_ROOT}"
for rank in "${!GPUS[@]}"; do
  merge_shards "${rank}"
done

final_count="$(sample_count)"
shard_count="$(find "${LMDB_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | wc -l)"
echo "Merged last-step-skip LoRA LMDB ready: ${LMDB_ROOT}"
echo "  shards : ${shard_count}"
echo "  samples: ${final_count}"

if (( final_count != TOTAL_SAMPLES )); then
  echo "Expected ${TOTAL_SAMPLES} samples, got ${final_count}" >&2
  exit 1
fi
