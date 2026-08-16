#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

TOTAL_PROMPTS="${TOTAL_PROMPTS:-2000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEEDS="${SEEDS:-42 100 2024}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_2k}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/oracle_dataset_4gpu}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-6}"
EXTRACT_T5="${EXTRACT_T5:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_VIDEOS="${CLEAN_VIDEOS:-0}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
if (( NUM_GPUS < 1 )); then
  echo "GPU_IDS is empty" >&2
  exit 2
fi

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Prompts file not found: ${PROMPTS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
T5_DIR="${OUT_ROOT}/t5_embeddings"
PARTS_DIR="${OUT_ROOT}/_parts"
mkdir -p "${T5_DIR}" "${PARTS_DIR}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
TOTAL_TRAJECTORIES=$((TOTAL_PROMPTS * NUM_SEEDS))

echo "================================================================================"
echo " 4-GPU Oracle Trajectory Dataset Build"
echo "================================================================================"
echo "  Project Root      : ${PROJECT_ROOT}"
echo "  Prompts File      : ${PROMPTS_FILE}"
echo "  Total Prompts     : ${TOTAL_PROMPTS}"
echo "  Seeds per Prompt  : ${SEEDS} (${NUM_SEEDS} seeds)"
echo "  Total Trajectories: ${TOTAL_TRAJECTORIES}"
echo "  GPU Devices       : ${GPU_IDS} (${NUM_GPUS} cards)"
echo "  Output Directory  : ${OUT_ROOT}"
echo "  Log Directory     : ${LOG_DIR}"
echo "  Extract T5 First  : ${EXTRACT_T5}"
echo "  Dry Run Mode      : ${DRY_RUN}"
echo "================================================================================"

# ── Step 1: Precompute Frozen T5 Embeddings ──────────────────────────────────
if [[ "${EXTRACT_T5}" == "1" ]]; then
  echo "[Step 1/3] Extracting frozen T5 embeddings & token sequences for ${TOTAL_PROMPTS} prompts..."
  python "${SCRIPT_DIR}/extract_prompt_t5_embeddings.py" \
    --prompts_file "${PROMPTS_FILE}" \
    --out_dir "${T5_DIR}" \
    --limit "${TOTAL_PROMPTS}" \
    --device "cuda:${GPUS[0]}" || {
      echo "T5 extraction on GPU failed, falling back to CPU or checking error..." >&2
  }
fi

# ── Step 2: Launch 4-GPU Workers ─────────────────────────────────────────────
echo "[Step 2/3] Launching ${NUM_GPUS} parallel GPU workers..."

base_count=$((TOTAL_PROMPTS / NUM_GPUS))
remainder=$((TOTAL_PROMPTS % NUM_GPUS))
offset=0
pids=()
worker_names=()
worker_logs=()

cleanup_workers() {
  if (( ${#pids[@]} > 0 )); then
    echo "Caught interrupt signal. Stopping active GPU workers..." >&2
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
  part_out="${PARTS_DIR}/${part_name}"
  log_path="${LOG_DIR}/${part_name}.log"

  echo "  --> Launch worker [${part_name}]: GPU=${gpu}, Prompt Offset=${offset}, Count=${count}"
  (
    cd "${PROJECT_ROOT}"
    GPU_ID="${gpu}" \
    PROMPT_OFFSET="${offset}" \
    LIMIT="${count}" \
    SEEDS="${SEEDS}" \
    PROMPTS_FILE="${PROMPTS_FILE}" \
    OUT_ROOT="${part_out}" \
    T5_EMBED_DIR="${T5_DIR}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    DRY_RUN="${DRY_RUN}" \
    CLEAN_VIDEOS="${CLEAN_VIDEOS}" \
    bash "${SCRIPT_DIR}/run_oracle_worker.sh"
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

echo "Workers active. Live tail status every ${MONITOR_INTERVAL}s..."

while (( remaining > 0 )); do
  sleep "${MONITOR_INTERVAL}"
  echo "===== Worker Status Check $(date '+%F %T') ====="
  for i in "${!pids[@]}"; do
    if (( finished[i] == 1 )); then
      continue
    fi
    pid="${pids[$i]}"
    name="${worker_names[$i]}"
    log_path="${worker_logs[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[RUNNING] ${name} (PID: ${pid})"
      if [[ -f "${log_path}" ]]; then
        tail -n "${MONITOR_TAIL_LINES}" "${log_path}" || true
      fi
    else
      if wait "${pid}"; then
        echo "[SUCCESS] ${name} finished successfully."
      else
        echo "[ERROR] ${name} failed! Last log lines:"
        if [[ -f "${log_path}" ]]; then
          tail -n 60 "${log_path}" || true
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
  echo "One or more workers encountered errors. Check logs under: ${LOG_DIR}" >&2
  exit 1
fi

# ── Step 3: Merge and Verify Dataset ─────────────────────────────────────────
echo "[Step 3/3] Merging worker shard parts and computing variance / regret metrics..."

python "${SCRIPT_DIR}/merge_and_verify_oracle_dataset.py" \
  --parts_dir "${PARTS_DIR}" \
  --out_root "${OUT_ROOT}" \
  --total_prompts "${TOTAL_PROMPTS}" \
  --seeds ${SEEDS}

echo "================================================================================"
echo " Dataset Generation Complete!"
echo " Master Manifest : ${OUT_ROOT}/dataset_manifest.json"
echo " Summary CSV     : ${OUT_ROOT}/dataset_summary.csv"
echo "================================================================================"
