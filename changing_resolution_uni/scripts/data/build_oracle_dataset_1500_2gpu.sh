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

GPU_IDS="${GPU_IDS:-0,1}"
BASE_PROMPT_OFFSET="${BASE_PROMPT_OFFSET:-0}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-1000}"
VAL_PROMPTS="${VAL_PROMPTS:-200}"
TEST_PROMPTS="${TEST_PROMPTS:-300}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-42 100 2024}"
TRAIN_INCLUDE_NATIVE_HR="${TRAIN_INCLUDE_NATIVE_HR:-1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
PROMPTS_FILE="${PROMPTS_FILE:-${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_VIDEOS="${CLEAN_VIDEOS:-0}"
SAVE_LATENTS="${SAVE_LATENTS:-1}"
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE:-fp16}"
DRY_RUN="${DRY_RUN:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-6}"
NUM_CANONICAL_PARTS=8

if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRACT_T5="${EXTRACT_T5:-0}"
else
  EXTRACT_T5="${EXTRACT_T5:-1}"
fi

for name in BASE_PROMPT_OFFSET TRAIN_PROMPTS VAL_PROMPTS TEST_PROMPTS; do
  value="${!name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 2
  fi
done

TOTAL_PROMPTS=$((TRAIN_PROMPTS + VAL_PROMPTS + TEST_PROMPTS))
if (( TOTAL_PROMPTS != 1500 )); then
  echo "This formal launcher requires exactly 1500 prompts; configured total=${TOTAL_PROMPTS}." >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_ACTIVE_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_ACTIVE_GPUS < 1 )); then
  echo "GPU_IDS must contain at least 1 GPU device, got: ${GPU_IDS}" >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ -z "${gpu}" || -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU_IDS must contain unique, non-empty device ids: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu}"]=1
done

read -r -a TRAIN_SEED_ARRAY <<< "${TRAIN_SEEDS}"
read -r -a EVAL_SEED_ARRAY <<< "${EVAL_SEEDS}"

TRAIN_OFFSET="${BASE_PROMPT_OFFSET}"
EVAL_OFFSET=$((BASE_PROMPT_OFFSET + TRAIN_PROMPTS))
VAL_OFFSET="${EVAL_OFFSET}"
TEST_OFFSET=$((VAL_OFFSET + VAL_PROMPTS))
END_OFFSET=$((BASE_PROMPT_OFFSET + TOTAL_PROMPTS))
CANDIDATE_COUNT=13
TRAIN_VIDEOS_PER_PROMPT=$((CANDIDATE_COUNT + TRAIN_INCLUDE_NATIVE_HR))
EVAL_VIDEOS_PER_PROMPT=$((CANDIDATE_COUNT + 1))
EXPECTED_TRAIN_VIDEOS=$((TRAIN_PROMPTS * ${#TRAIN_SEED_ARRAY[@]} * TRAIN_VIDEOS_PER_PROMPT))
EXPECTED_EVAL_VIDEOS=$(((VAL_PROMPTS + TEST_PROMPTS) * ${#EVAL_SEED_ARRAY[@]} * EVAL_VIDEOS_PER_PROMPT))
EXPECTED_TOTAL_VIDEOS=$((EXPECTED_TRAIN_VIDEOS + EXPECTED_EVAL_VIDEOS))
TRAIN_TRAJECTORIES=$((TRAIN_PROMPTS * ${#TRAIN_SEED_ARRAY[@]}))
EVAL_TRAJECTORIES=$(((VAL_PROMPTS + TEST_PROMPTS) * ${#EVAL_SEED_ARRAY[@]}))
EXPECTED_LATENT_FILES=$(((TRAIN_TRAJECTORIES + EVAL_TRAJECTORIES) * CANDIDATE_COUNT))
if [[ "${LATENT_SAVE_DTYPE}" == "fp32" ]]; then
  LATENT_BYTES_PER_VALUE=4
else
  LATENT_BYTES_PER_VALUE=2
fi
ESTIMATED_LATENT_BYTES=$((EXPECTED_LATENT_FILES * 2 * 16 * 21 * 46 * 80 * LATENT_BYTES_PER_VALUE))
ESTIMATED_LATENT_GIB=$(((ESTIMATED_LATENT_BYTES + 1073741823) / 1073741824))

echo "================================================================================"
echo " 1500-Prompt Dynamic Worker Breakpoint Resume Plan"
echo "================================================================================"
echo " Active GPUs : ${GPU_IDS} (${NUM_ACTIVE_GPUS} cards running ${NUM_CANONICAL_PARTS} canonical parts)"
echo " Prompt file : ${PROMPTS_FILE}"
echo " Output root : ${OUT_ROOT}"
echo " Train       : prompts [${TRAIN_OFFSET}, $((EVAL_OFFSET - 1))], count=${TRAIN_PROMPTS}, seeds='${TRAIN_SEEDS}'"
echo " Validation  : prompts [${VAL_OFFSET}, $((TEST_OFFSET - 1))], count=${VAL_PROMPTS}, seeds='${EVAL_SEEDS}'"
echo " Test        : prompts [${TEST_OFFSET}, $((END_OFFSET - 1))], count=${TEST_PROMPTS}, seeds='${EVAL_SEEDS}'"
echo " Expected vids: train=${EXPECTED_TRAIN_VIDEOS}, eval=${EXPECTED_EVAL_VIDEOS}, total=${EXPECTED_TOTAL_VIDEOS}"
echo " Shard Layout: Preserving canonical 8-part layout (_parts/part_00 .. part_07)"
echo " Resume/skip : ${SKIP_EXISTING} (Skip finished prompts in milliseconds)"
echo "================================================================================"

if [[ "${PLAN_ONLY}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUT_ROOT}"
PLAN_PATH="${OUT_ROOT}/generation_plan.json"

# Write / ensure generation plan exists
if [[ ! -f "${PLAN_PATH}" ]]; then
  python - "${PLAN_PATH}" "${GPU_IDS}" "${PROMPTS_FILE}" "${BASE_PROMPT_OFFSET}" \
    "${TRAIN_PROMPTS}" "${VAL_PROMPTS}" "${TEST_PROMPTS}" "${TRAIN_SEEDS}" \
    "${EVAL_SEEDS}" "${TRAIN_INCLUDE_NATIVE_HR}" "${EXPECTED_TRAIN_VIDEOS}" \
    "${EXPECTED_EVAL_VIDEOS}" "${EXPECTED_TOTAL_VIDEOS}" "${PRIMARY_LAMBDA}" \
    "${EXPECTED_LATENT_FILES}" "${LATENT_SAVE_DTYPE}" "${ESTIMATED_LATENT_BYTES}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

(
    output,
    gpu_ids,
    prompts_file,
    base_offset,
    train_prompts,
    val_prompts,
    test_prompts,
    train_seeds,
    eval_seeds,
    train_native,
    train_videos,
    eval_videos,
    total_videos,
    primary_lambda,
    latent_files,
    latent_dtype,
    latent_bytes,
) = sys.argv[1:]
base = int(base_offset)
train_count = int(train_prompts)
val_count = int(val_prompts)
test_count = int(test_prompts)
payload = {
    "schema": "oracle_1500_8gpu_generation_plan_v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "gpu_ids": gpu_ids.split(","),
    "prompts_file": str(Path(prompts_file).expanduser()),
    "candidate_steps": [30, 35, *range(40, 51)],
    "primary_lambda": float(primary_lambda),
    "splits": {
        "train": {
            "prompt_offset": base,
            "prompt_count": train_count,
            "seeds": [int(v) for v in train_seeds.split()],
            "include_native_hr": bool(int(train_native)),
            "expected_videos": int(train_videos),
            "physical_dataset": "train",
        },
        "validation": {
            "prompt_offset": base + train_count,
            "prompt_count": val_count,
            "seeds": [int(v) for v in eval_seeds.split()],
            "include_native_hr": True,
            "physical_dataset": "eval",
        },
        "test": {
            "prompt_offset": base + train_count + val_count,
            "prompt_count": test_count,
            "seeds": [int(v) for v in eval_seeds.split()],
            "include_native_hr": True,
            "physical_dataset": "eval",
        },
    },
    "expected_videos": {
        "train": int(train_videos),
        "eval": int(eval_videos),
        "total": int(total_videos),
    },
    "artifacts": {
        "preserve_candidate_videos": True,
        "save_switch_latents": True,
        "latent_schema": "wan_taa_free_oracle_latent_v1",
        "latent_tensors": ["x_t_lr", "x0_pred_lr"],
        "latent_dtype": latent_dtype,
        "expected_latent_files": int(latent_files),
        "estimated_latent_bytes": int(latent_bytes),
    },
    "strict_vbench_ready": {
        "train": bool(int(train_native)),
        "validation": True,
        "test": True,
    },
}
Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
fi

run_split_dynamic_queue() {
  local split_name="$1"
  local base_offset="$2"
  local total_split_prompts="$3"
  local seeds="$4"
  local include_native="$5"

  local split_root="${OUT_ROOT}/${split_name}"
  local split_logs="${OUT_ROOT}/logs/${split_name}"
  local parts_dir="${split_root}/_parts"
  local t5_dir="${split_root}/t5_embeddings"
  local queue_claims_dir="${split_root}/.queue_claims"

  mkdir -p "${split_root}" "${split_logs}" "${parts_dir}" "${t5_dir}"
  rm -rf "${queue_claims_dir}"
  mkdir -p "${queue_claims_dir}"

  echo "================================================================================"
  echo " Starting split '${split_name}': ${total_split_prompts} prompts across ${NUM_CANONICAL_PARTS} canonical parts"
  echo " Active GPU workers: ${NUM_ACTIVE_GPUS} cards (${GPU_IDS})"
  echo "================================================================================"

  # Step 1: Pre-extract T5 embeddings for this split
  if [[ "${EXTRACT_T5}" == "1" ]]; then
    echo "[Step 1/2] Pre-checking / extracting T5 embeddings (offset=${base_offset}, limit=${total_split_prompts})..."
    python "${SCRIPT_DIR}/extract_prompt_t5_embeddings.py" \
      --prompts_file "${PROMPTS_FILE}" \
      --out_dir "${t5_dir}" \
      --prompt_offset "${base_offset}" \
      --limit "${total_split_prompts}" \
      --device "cuda:${GPU_ARRAY[0]}" --skip_existing || {
        echo "T5 GPU extraction failed, falling back to CPU..." >&2
        python "${SCRIPT_DIR}/extract_prompt_t5_embeddings.py" \
          --prompts_file "${PROMPTS_FILE}" \
          --out_dir "${t5_dir}" \
          --prompt_offset "${base_offset}" \
          --limit "${total_split_prompts}" \
          --device "cpu" --skip_existing
      }
  fi

  # Compute canonical 8-part ranges
  local base_count=$((total_split_prompts / NUM_CANONICAL_PARTS))
  local remainder=$((total_split_prompts % NUM_CANONICAL_PARTS))
  local part_offsets=()
  local part_counts=()
  local curr_offset="${base_offset}"

  for ((p=0; p<NUM_CANONICAL_PARTS; p++)); do
    local c="${base_count}"
    if (( p < remainder )); then
      c=$((c + 1))
    fi
    part_offsets+=("${curr_offset}")
    part_counts+=("${c}")
    curr_offset=$((curr_offset + c))
  done

  # Step 2: Dynamic FIFO Worker loop across active GPUs
  echo "[Step 2/2] Launching ${NUM_ACTIVE_GPUS} dynamic GPU workers..."

  worker_pids=()
  worker_gpus=()

  cleanup_dynamic_workers() {
    if (( ${#worker_pids[@]} > 0 )); then
      echo "Stopping active GPU workers and child processes..." >&2
      for pid in "${worker_pids[@]}"; do
        pkill -P "${pid}" 2>/dev/null || true
        kill "${pid}" 2>/dev/null || true
      done
    fi
  }
  trap 'cleanup_dynamic_workers; exit 130' INT TERM

  for gpu_idx in "${!GPU_ARRAY[@]}"; do
    local gpu="${GPU_ARRAY[$gpu_idx]}"
    local worker_id="worker_${gpu_idx}_gpu${gpu}"
    local worker_main_log="${split_logs}/${worker_id}.log"

    echo "  --> Spawned dynamic worker [${worker_id}] on GPU ${gpu}"
    (
      cd "${PROJECT_ROOT}"
      for ((p_idx=0; p_idx<NUM_CANONICAL_PARTS; p_idx++)); do
        local claim_file="${queue_claims_dir}/part_${p_idx}"
        # Atomic claim via mkdir
        if mkdir "${claim_file}" 2>/dev/null; then
          local p_name="$(printf "part_%02d" "${p_idx}")"
          local p_offset="${part_offsets[$p_idx]}"
          local p_count="${part_counts[$p_idx]}"
          local p_out="${parts_dir}/${p_name}"
          local p_log="${split_logs}/${p_name}.log"

          echo "[$(date '+%T')] [GPU ${gpu}] Claimed ${p_name} (offset=${p_offset}, count=${p_count}). Starting..."
          
          if ! GPU_ID="${gpu}" \
            PROMPT_OFFSET="${p_offset}" \
            LIMIT="${p_count}" \
            SEEDS="${seeds}" \
            PROMPTS_FILE="${PROMPTS_FILE}" \
            OUT_ROOT="${p_out}" \
            T5_EMBED_DIR="${t5_dir}" \
            SKIP_EXISTING="${SKIP_EXISTING}" \
            DRY_RUN="${DRY_RUN}" \
            CLEAN_VIDEOS="${CLEAN_VIDEOS}" \
            INCLUDE_NATIVE_HR="${include_native}" \
            SAVE_LATENTS="${SAVE_LATENTS}" \
            LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE}" \
            PRIMARY_LAMBDA="${PRIMARY_LAMBDA}" \
            bash "${SCRIPT_DIR}/run_oracle_worker.sh" >"${p_log}" 2>&1; then
            echo "[$(date '+%T')] [GPU ${gpu}] ERROR: Worker failed on ${p_name}! Check log: ${p_log}" >&2
            exit 1
          fi

          echo "[$(date '+%T')] [GPU ${gpu}] Finished ${p_name}."
        fi
      done
    ) >"${worker_main_log}" 2>&1 &

    worker_pids+=("$!")
    worker_gpus+=("${gpu}")
  done

  # Monitor status
  local failed=0
  local remaining="${#worker_pids[@]}"
  local finished_workers=()
  for _ in "${worker_pids[@]}"; do
    finished_workers+=(0)
  done

  while (( remaining > 0 )); do
    sleep "${MONITOR_INTERVAL}"
    echo "===== [${split_name}] Status Check $(date '+%F %T') ====="
    for i in "${!worker_pids[@]}"; do
      if (( finished_workers[i] == 1 )); then
        continue
      fi
      local pid="${worker_pids[$i]}"
      local gpu="${worker_gpus[$i]}"
      if kill -0 "${pid}" 2>/dev/null; then
        echo "[RUNNING] Worker GPU ${gpu} (PID: ${pid})"
      else
        if wait "${pid}"; then
          echo "[SUCCESS] Worker GPU ${gpu} finished all assigned parts."
        else
          echo "[ERROR] Worker GPU ${gpu} encountered errors!"
          failed=1
        fi
        finished_workers[i]=1
        remaining=$((remaining - 1))
      fi
    done

    # Print summary of part completions
    for ((p_idx=0; p_idx<NUM_CANONICAL_PARTS; p_idx++)); do
      local p_name="$(printf "part_%02d" "${p_idx}")"
      local p_log="${split_logs}/${p_name}.log"
      if [[ -f "${p_log}" ]]; then
        local last_line
        last_line="$(tail -n 1 "${p_log}" 2>/dev/null || true)"
        echo "  - ${p_name}: ${last_line}"
      fi
    done
  done

  trap - INT TERM

  if (( failed != 0 )); then
    echo "One or more workers failed during split '${split_name}'. Check logs in ${split_logs}" >&2
    exit 1
  fi

  # Merge split parts
  echo "Merging worker shards for split '${split_name}'..."
  python "${SCRIPT_DIR}/merge_and_verify_oracle_dataset.py" \
    --parts_dir "${parts_dir}" \
    --out_root "${split_root}" \
    --total_prompts "${total_split_prompts}" \
    --seeds ${seeds} \
    --primary_lambda "${PRIMARY_LAMBDA}"
}

# Run Train (1000 prompts, 8 parts) then Eval (500 prompts, 8 parts)
run_split_dynamic_queue "train" "${TRAIN_OFFSET}" "${TRAIN_PROMPTS}" "${TRAIN_SEEDS}" "${TRAIN_INCLUDE_NATIVE_HR}"
run_split_dynamic_queue "eval" "${EVAL_OFFSET}" "$((VAL_PROMPTS + TEST_PROMPTS))" "${EVAL_SEEDS}" "1"

# Final integrity check
python - "${OUT_ROOT}/generation_complete.json" "${PLAN_PATH}" \
  "${OUT_ROOT}/train/dataset_manifest.json" "${OUT_ROOT}/eval/dataset_manifest.json" \
  "${OUT_ROOT}/train" "${OUT_ROOT}/eval" "${DRY_RUN}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
train_path = Path(sys.argv[3])
eval_path = Path(sys.argv[4])
train_root = Path(sys.argv[5])
eval_root = Path(sys.argv[6])
dry_run = bool(int(sys.argv[7]))
plan = json.loads(plan_path.read_text(encoding="utf-8"))
manifests = {}
for name, path in (("train", train_path), ("eval", eval_path)):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("is_complete") is not True:
        raise SystemExit(f"{name} dataset is incomplete: {path}")
    manifests[name] = {
        "path": str(path),
        "total_prompts_found": payload.get("total_prompts_found"),
        "total_trajectories": payload.get("total_trajectories"),
        "is_complete": True,
    }

candidate_count = len(plan["candidate_steps"])
train_spec = plan["splits"]["train"]
val_spec = plan["splits"]["validation"]
test_spec = plan["splits"]["test"]
train_trajectories = train_spec["prompt_count"] * len(train_spec["seeds"])
eval_trajectories = (
    val_spec["prompt_count"] * len(val_spec["seeds"])
    + test_spec["prompt_count"] * len(test_spec["seeds"])
)

def nonempty_files(root: Path, pattern: str) -> list[Path]:
    return [path for path in root.rglob(pattern) if path.is_file() and path.stat().st_size > 0]

coverage = {}
for name, root, trajectories, expect_native in (
    ("train", train_root, train_trajectories, train_spec["include_native_hr"]),
    ("eval", eval_root, eval_trajectories, True),
):
    candidate_videos = nonempty_files(root, "videos/step*/*.mp4")
    native_videos = nonempty_files(root, "videos/native_hr/*.mp4")
    latent_files = nonempty_files(root, "latents/step*/*.pt")
    expected_candidate = trajectories * candidate_count
    expected_native = trajectories if expect_native else 0
    expected_latents = trajectories * candidate_count
    split_coverage = {
        "candidate_videos": len(candidate_videos),
        "expected_candidate_videos": expected_candidate,
        "native_videos": len(native_videos),
        "expected_native_videos": expected_native,
        "switch_latents": len(latent_files),
        "expected_switch_latents": expected_latents,
        "nonempty_files_only": True,
        "verified": dry_run or (
            len(candidate_videos) == expected_candidate
            and len(native_videos) == expected_native
            and len(latent_files) == expected_latents
        ),
    }
    if not split_coverage["verified"]:
        raise SystemExit(
            f"{name} artifact coverage mismatch: {json.dumps(split_coverage, sort_keys=True)}"
        )
    coverage[name] = split_coverage

result = {
    "schema": "oracle_1500_8gpu_generation_complete_v1",
    "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "plan": str(plan_path),
    "expected_videos": plan["expected_videos"],
    "manifests": manifests,
    "artifact_coverage": coverage,
    "artifact_coverage_skipped_for_dry_run": dry_run,
}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "================================================================================"
echo " 1500-Prompt Oracle Dataset Generation Complete!"
echo " Plan      : ${PLAN_PATH}"
echo " Completion: ${OUT_ROOT}/generation_complete.json"
echo "================================================================================"
