#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
PROMPTS_FILE="${PROMPTS_FILE:-${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}}"
MICRO_BATCH_PROMPTS="${MICRO_BATCH_PROMPTS:-2}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE:-fp16}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
PLAN_ONLY="${PLAN_ONLY:-0}"
CONFIRM_EXCLUSIVE="${CONFIRM_EXCLUSIVE:-0}"
CANDIDATE_STEPS=(30 35 40 41 42 43 44 45 46 47 48 49 50)

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${RUN_ID}" ]] || die "RUN_ID is required"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "RUN_ID contains unsafe characters"
[[ "${MICRO_BATCH_PROMPTS}" =~ ^[1-9][0-9]*$ ]] || die "MICRO_BATCH_PROMPTS must be positive"
[[ "${MONITOR_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "MONITOR_INTERVAL must be positive"
[[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || die "MIN_FREE_GIB must be non-negative"
[[ "${LATENT_SAVE_DTYPE}" =~ ^(fp16|bf16|fp32)$ ]] || die "invalid LATENT_SAVE_DTYPE"
[[ -d "${OUT_ROOT}" ]] || die "existing OUT_ROOT not found: ${OUT_ROOT}"
[[ -f "${PROMPTS_FILE}" ]] || die "PROMPTS_FILE not found: ${PROMPTS_FILE}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
(( ${#GPU_ARRAY[@]} == 8 )) || die "H100 tail launcher requires exactly 8 GPU IDs"
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU IDs must be numeric"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "GPU IDs must be unique"
  SEEN_GPUS["${gpu}"]=1
done

ORCH_ROOT="${OUT_ROOT}/.orchestration/h100_tail/${RUN_ID}"
PLAN_PATH="${ORCH_ROOT}/tail_plan.json"
LOG_DIR="${OUT_ROOT}/logs/h100_tail/${RUN_ID}"
mkdir -p "${ORCH_ROOT}" "${LOG_DIR}"

python "${SCRIPT_DIR}/plan_oracle_h100_tail.py" \
  --out-root "${OUT_ROOT}" \
  --prompts-file "${PROMPTS_FILE}" \
  --plan-out "${PLAN_PATH}" \
  --micro-batch-prompts "${MICRO_BATCH_PROMPTS}"

if [[ "${PLAN_ONLY}" == "1" ]]; then
  echo "Tail plan: ${PLAN_PATH}"
  exit 0
fi

[[ "${CONFIRM_EXCLUSIVE}" == "1" ]] || die "set CONFIRM_EXCLUSIVE=1 only after all previous A800/H100 writers have stopped"

free_kib="$(df -Pk "${OUT_ROOT}" | awk 'NR==2 {print $4}')"
[[ "${free_kib}" =~ ^[0-9]+$ ]] || die "could not determine free space"
free_gib=$((free_kib / 1024 / 1024))
(( free_gib >= MIN_FREE_GIB )) || die "only ${free_gib} GiB free; require ${MIN_FREE_GIB} GiB"

python - "${OUT_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for split, offset, count in (("train", 0, 1000), ("eval", 1000, 500)):
    t5 = root / split / "t5_embeddings"
    for prompt_id in range(offset, offset + count):
        npz = t5 / f"prompt_{prompt_id:06d}.npz"
        meta = t5 / f"prompt_{prompt_id:06d}.json"
        if not npz.is_file() or npz.stat().st_size == 0 or not meta.is_file() or meta.stat().st_size == 0:
            raise SystemExit(f"T5 coverage incomplete: {split}/prompt_{prompt_id:06d}")
        payload = json.loads(meta.read_text(encoding="utf-8"))
        if int(payload.get("prompt_id", -1)) != prompt_id:
            raise SystemExit(f"T5 metadata mismatch: {meta}")
PY

mapfile -t TASK_LINES < <(python - "${PLAN_PATH}" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for task in plan["tasks"]:
    print("\t".join(str(task[key]) for key in (
        "task_id", "split", "part", "base_seed", "prompt_offset", "limit",
        "canonical_prompt_offset", "canonical_prompt_count"
    )))
PY
)

echo "================================================================================"
echo " H100 tail run: ${#TASK_LINES[@]} exact missing micro-tasks on 8 GPUs"
echo " Plan          : ${PLAN_PATH}"
echo " Micro-batch   : ${MICRO_BATCH_PROMPTS} prompts/task, one seed/task"
echo " Existing data : retained in place"
echo "================================================================================"

ATTEMPT_ID="$(hostname | tr -cd 'A-Za-z0-9_.-')_$(date -u +%Y%m%dT%H%M%SZ)_$$"
CLAIMS_DIR="${ORCH_ROOT}/attempts/${ATTEMPT_ID}/claims"
DONE_DIR="${ORCH_ROOT}/attempts/${ATTEMPT_ID}/done"
mkdir -p "${CLAIMS_DIR}" "${DONE_DIR}"

worker_pids=()
cleanup_workers() {
  local pid
  for pid in "${worker_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}

if (( ${#TASK_LINES[@]} > 0 )); then
  for worker_idx in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$worker_idx]}"
    worker_log="${LOG_DIR}/worker_${worker_idx}_gpu${gpu}.log"
    (
      for line in "${TASK_LINES[@]}"; do
        IFS=$'\t' read -r task_id split part seed prompt_offset limit canonical_offset canonical_count <<< "${line}"
        claim="${CLAIMS_DIR}/${task_id}"
        if ! mkdir "${claim}" 2>/dev/null; then
          continue
        fi
        part_name="$(printf 'part_%02d' "${part}")"
        part_root="${OUT_ROOT}/${split}/_parts/${part_name}"
        t5_dir="${OUT_ROOT}/${split}/t5_embeddings"
        task_log="${LOG_DIR}/${task_id}.log"
        marker="${DONE_DIR}/${task_id}.done.json"
        echo "[$(date '+%F %T')] GPU ${gpu} claimed ${task_id}"
        GPU_ID="${gpu}" \
          PROMPT_OFFSET="${prompt_offset}" \
          LIMIT="${limit}" \
          PROTOCOL_PROMPT_OFFSET="${canonical_offset}" \
          PROTOCOL_PROMPT_LIMIT="${canonical_count}" \
          SEEDS="${seed}" \
          PROMPTS_FILE="${PROMPTS_FILE}" \
          OUT_ROOT="${part_root}" \
          T5_EMBED_DIR="${t5_dir}" \
          SKIP_EXISTING=1 \
          CLEAN_VIDEOS=0 \
          INCLUDE_NATIVE_HR=1 \
          SAVE_LATENTS=1 \
          LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE}" \
          ENABLE_INLINE_VBENCH=0 \
          PRIMARY_LAMBDA="${PRIMARY_LAMBDA}" \
          bash "${SCRIPT_DIR}/run_oracle_worker.sh" >"${task_log}" 2>&1
        python "${SCRIPT_DIR}/verify_oracle_resume_task.py" \
          --part-root "${part_root}" \
          --prompt-offset "${prompt_offset}" \
          --limit "${limit}" \
          --seeds "${seed}" \
          --candidate-steps "${CANDIDATE_STEPS[@]}" \
          --include-native-hr 1 \
          --require-latents 1 \
          --ignore-records \
          --marker "${marker}" \
          --quiet
        echo "[$(date '+%F %T')] GPU ${gpu} completed ${task_id}"
      done
    ) >"${worker_log}" 2>&1 &
    worker_pids+=("$!")
  done

  trap 'cleanup_workers; exit 130' INT TERM
  remaining="${#worker_pids[@]}"
  failed=0
  finished=()
  for _ in "${worker_pids[@]}"; do finished+=(0); done
  while (( remaining > 0 )); do
    sleep "${MONITOR_INTERVAL}"
    for i in "${!worker_pids[@]}"; do
      (( finished[i] == 0 )) || continue
      pid="${worker_pids[$i]}"
      if ! kill -0 "${pid}" 2>/dev/null; then
        if ! wait "${pid}"; then failed=1; fi
        finished[i]=1
        remaining=$((remaining - 1))
      fi
    done
    done_count="$(find "${DONE_DIR}" -maxdepth 1 -name '*.done.json' -type f | wc -l)"
    echo "[$(date '+%F %T')] H100 tail: ${done_count}/${#TASK_LINES[@]} tasks done; ${remaining} workers running"
    if (( failed != 0 )); then
      cleanup_workers
      wait || true
      trap - INT TERM
      die "one or more H100 tail workers failed; inspect ${LOG_DIR}"
    fi
  done
  trap - INT TERM
fi

python "${SCRIPT_DIR}/plan_oracle_h100_tail.py" \
  --out-root "${OUT_ROOT}" \
  --prompts-file "${PROMPTS_FILE}" \
  --plan-out "${ORCH_ROOT}/post_generation_plan.json" \
  --micro-batch-prompts "${MICRO_BATCH_PROMPTS}" \
  --require-complete

split_part_range() {
  local split_offset="$1" split_count="$2" part="$3"
  local base_count=$((split_count / 8)) remainder=$((split_count % 8))
  local count="${base_count}"
  local offset=$((split_offset + part * base_count + (part < remainder ? part : remainder)))
  if (( part < remainder )); then count=$((count + 1)); fi
  echo "${offset} ${count}"
}

echo "Packaging/repairing trajectory records without GPU generation..."
for split in train eval; do
  if [[ "${split}" == "train" ]]; then
    split_offset=0; split_count=1000; seeds="42"
  else
    split_offset=1000; split_count=500; seeds="42 100 2024"
  fi
  for part in {0..7}; do
    read -r offset count < <(split_part_range "${split_offset}" "${split_count}" "${part}")
    part_name="$(printf 'part_%02d' "${part}")"
    python "${SCRIPT_DIR}/build_oracle_trajectory_dataset.py" \
      --prompts_file "${PROMPTS_FILE}" \
      --out_root "${OUT_ROOT}/${split}/_parts/${part_name}" \
      --prompt_offset "${offset}" \
      --limit "${count}" \
      --seeds ${seeds} \
      --candidate_steps "${CANDIDATE_STEPS[@]}" \
      --infer_steps 50 \
      --primary_lambda "${PRIMARY_LAMBDA}" \
      --t5_embed_dir "${OUT_ROOT}/${split}/t5_embeddings" \
      --skip_existing
  done
done

echo "Running the standard single-node verifier, merge, and final coverage gate..."
NODE_RANK=0 \
NUM_NODES=1 \
RUN_ID="${RUN_ID}_finalize" \
NODE_NAME=h100_tail_finalize \
PART_INDICES=0,1,2,3,4,5,6,7 \
GPU_IDS="${GPU_IDS}" \
OUT_ROOT="${OUT_ROOT}" \
PROMPTS_FILE="${PROMPTS_FILE}" \
PRIMARY_LAMBDA="${PRIMARY_LAMBDA}" \
SKIP_EXISTING=1 \
CLEAN_VIDEOS=0 \
SAVE_LATENTS=1 \
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE}" \
EXTRACT_T5=0 \
MIN_FREE_GIB=0 \
bash "${SCRIPT_DIR}/build_oracle_dataset_1500_2gpu.sh"

echo "H100 tail completion verified: ${OUT_ROOT}/generation_complete.json"
