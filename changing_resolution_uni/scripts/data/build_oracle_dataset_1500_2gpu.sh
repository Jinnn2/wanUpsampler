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
PART_INDICES="${PART_INDICES:-0,1,2,3,4,5,6,7}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
RUN_ID="${RUN_ID:-single_node}"
NODE_NAME="${NODE_NAME:-node${NODE_RANK}}"
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
EXTRACT_T5="${EXTRACT_T5:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-604800}"
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"
NUM_CANONICAL_PARTS=8
LANES_PER_PART=2
NUM_RESUME_TASKS=$((NUM_CANONICAL_PARTS * LANES_PER_PART))
CANDIDATE_STEPS=(30 35 40 41 42 43 44 45 46 47 48 49 50)

die() {
  echo "ERROR: $*" >&2
  exit 2
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

for name in BASE_PROMPT_OFFSET TRAIN_PROMPTS VAL_PROMPTS TEST_PROMPTS NODE_RANK NUM_NODES \
  MONITOR_INTERVAL WAIT_INTERVAL WAIT_TIMEOUT MIN_FREE_GIB; do
  value="${!name}"
  is_nonnegative_integer "${value}" || die "${name} must be a non-negative integer, got: ${value}"
done
(( NUM_NODES == 1 || NUM_NODES == 2 )) || die "NUM_NODES must be 1 or 2"
(( NODE_RANK < NUM_NODES )) || die "NODE_RANK=${NODE_RANK} must be smaller than NUM_NODES=${NUM_NODES}"
(( MONITOR_INTERVAL > 0 && WAIT_INTERVAL > 0 )) || die "monitor/wait intervals must be positive"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "RUN_ID may contain only letters, digits, dot, underscore, and hyphen"
[[ "${NODE_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "NODE_NAME may contain only letters, digits, dot, underscore, and hyphen"

TOTAL_PROMPTS=$((TRAIN_PROMPTS + VAL_PROMPTS + TEST_PROMPTS))
(( TRAIN_PROMPTS > 0 && VAL_PROMPTS > 0 && TEST_PROMPTS > 0 )) || die "all split sizes must be positive"
(( TOTAL_PROMPTS == 1500 )) || die "formal launcher requires exactly 1500 prompts; got ${TOTAL_PROMPTS}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
(( ${#GPU_ARRAY[@]} > 0 )) || die "GPU_IDS must not be empty"
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU_IDS must contain numeric device ids: ${GPU_IDS}"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "GPU_IDS must be unique: ${GPU_IDS}"
  SEEN_GPUS["${gpu}"]=1
done

IFS=',' read -r -a PART_ARRAY <<< "${PART_INDICES}"
(( ${#PART_ARRAY[@]} > 0 )) || die "PART_INDICES must not be empty"
declare -A SEEN_PARTS=()
for part in "${PART_ARRAY[@]}"; do
  [[ "${part}" =~ ^[0-9]+$ ]] || die "PART_INDICES must contain integers: ${PART_INDICES}"
  (( part >= 0 && part < NUM_CANONICAL_PARTS )) || die "part index ${part} is outside [0,7]"
  [[ -z "${SEEN_PARTS[${part}]:-}" ]] || die "PART_INDICES must be unique: ${PART_INDICES}"
  SEEN_PARTS["${part}"]=1
done

read -r -a TRAIN_SEED_ARRAY <<< "${TRAIN_SEEDS}"
read -r -a EVAL_SEED_ARRAY <<< "${EVAL_SEEDS}"
(( ${#TRAIN_SEED_ARRAY[@]} == 1 )) || die "TRAIN_SEEDS must contain exactly one base seed"
(( ${#EVAL_SEED_ARRAY[@]} == 3 )) || die "EVAL_SEEDS must contain exactly three base seeds"
for seed in "${TRAIN_SEED_ARRAY[@]}" "${EVAL_SEED_ARRAY[@]}"; do
  is_nonnegative_integer "${seed}" || die "seeds must be non-negative integers"
done
[[ "${TRAIN_INCLUDE_NATIVE_HR}" == "1" ]] || die "TRAIN_INCLUDE_NATIVE_HR must remain 1 for strict continuation"
[[ "${SKIP_EXISTING}" == "1" ]] || die "SKIP_EXISTING must remain 1 for an in-place resume"
[[ "${CLEAN_VIDEOS}" == "0" ]] || die "CLEAN_VIDEOS must remain 0"
[[ "${SAVE_LATENTS}" == "1" ]] || die "SAVE_LATENTS must remain 1"
[[ "${LATENT_SAVE_DTYPE}" =~ ^(fp16|bf16|fp32)$ ]] || die "invalid LATENT_SAVE_DTYPE=${LATENT_SAVE_DTYPE}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${EXTRACT_T5}" == "0" || "${EXTRACT_T5}" == "1" ]] || die "EXTRACT_T5 must be 0 or 1"

if (( NUM_NODES == 2 )); then
  (( ${#GPU_ARRAY[@]} == 8 )) || die "two-node mode requires exactly 8 GPUs per node"
  (( ${#PART_ARRAY[@]} == 4 )) || die "two-node mode requires exactly 4 canonical parts per node"
fi

TRAIN_OFFSET="${BASE_PROMPT_OFFSET}"
EVAL_OFFSET=$((BASE_PROMPT_OFFSET + TRAIN_PROMPTS))
VAL_OFFSET="${EVAL_OFFSET}"
TEST_OFFSET=$((VAL_OFFSET + VAL_PROMPTS))
END_OFFSET=$((BASE_PROMPT_OFFSET + TOTAL_PROMPTS))
EXPECTED_TRAIN_VIDEOS=$((TRAIN_PROMPTS * ${#TRAIN_SEED_ARRAY[@]} * 14))
EXPECTED_EVAL_VIDEOS=$(((VAL_PROMPTS + TEST_PROMPTS) * ${#EVAL_SEED_ARRAY[@]} * 14))
EXPECTED_TOTAL_VIDEOS=$((EXPECTED_TRAIN_VIDEOS + EXPECTED_EVAL_VIDEOS))
EXPECTED_LATENT_FILES=$(((TRAIN_PROMPTS * ${#TRAIN_SEED_ARRAY[@]} + (VAL_PROMPTS + TEST_PROMPTS) * ${#EVAL_SEED_ARRAY[@]}) * ${#CANDIDATE_STEPS[@]}))

echo "================================================================================"
echo " 1500-Prompt / 16-GPU Two-Node In-Place Resume Plan"
echo "================================================================================"
echo " Run ID       : ${RUN_ID}"
echo " Node          : rank=${NODE_RANK}/${NUM_NODES}, name=${NODE_NAME}"
echo " GPUs          : ${GPU_IDS} (${#GPU_ARRAY[@]} workers)"
echo " Canonical Parts: ${PART_INDICES} (${LANES_PER_PART} lanes per part)"
echo " Resume Tasks  : $((${#PART_ARRAY[@]} * LANES_PER_PART)) on this node / ${NUM_RESUME_TASKS} total"
echo " Output root   : ${OUT_ROOT}"
echo " Prompt file   : ${PROMPTS_FILE}"
echo " Train/Val/Test: ${TRAIN_PROMPTS}/${VAL_PROMPTS}/${TEST_PROMPTS}"
echo " Expected      : ${EXPECTED_TOTAL_VIDEOS} videos, ${EXPECTED_LATENT_FILES} latent files"
echo " Existing files: retained and checked in place; no migration"
echo "================================================================================"

split_part_range() {
  local split_offset="$1"
  local split_count="$2"
  local part="$3"
  local base_count=$((split_count / NUM_CANONICAL_PARTS))
  local remainder=$((split_count % NUM_CANONICAL_PARTS))
  local count="${base_count}"
  local offset=$((split_offset + part * base_count + (part < remainder ? part : remainder)))
  if (( part < remainder )); then count=$((count + 1)); fi
  echo "${offset} ${count}"
}

for part in "${PART_ARRAY[@]}"; do
  read -r train_part_offset train_part_count < <(split_part_range "${TRAIN_OFFSET}" "${TRAIN_PROMPTS}" "${part}")
  train_lane0=$(((train_part_count + 1) / 2))
  train_lane1=$((train_part_count - train_lane0))
  read -r eval_part_offset eval_part_count < <(split_part_range "${EVAL_OFFSET}" "$((VAL_PROMPTS + TEST_PROMPTS))" "${part}")
  eval_lane0=$(((eval_part_count + 1) / 2))
  eval_lane1=$((eval_part_count - eval_lane0))
  printf ' part_%02d -> train lanes [%d+%d, %d+%d], eval lanes [%d+%d, %d+%d]\n' \
    "${part}" "${train_part_offset}" "${train_lane0}" "$((train_part_offset + train_lane0))" "${train_lane1}" \
    "${eval_part_offset}" "${eval_lane0}" "$((eval_part_offset + eval_lane0))" "${eval_lane1}"
done

if [[ "${PLAN_ONLY}" == "1" ]]; then exit 0; fi

[[ -f "${PROMPTS_FILE}" ]] || die "prompt file not found: ${PROMPTS_FILE}"
mkdir -p "${OUT_ROOT}"
if [[ "${DRY_RUN}" != "1" && "${MIN_FREE_GIB}" != "0" ]]; then
  free_kib="$(df -Pk "${OUT_ROOT}" | awk 'NR==2 {print $4}')"
  is_nonnegative_integer "${free_kib}" || die "could not determine free disk space for ${OUT_ROOT}"
  free_gib=$((free_kib / 1024 / 1024))
  (( free_gib >= MIN_FREE_GIB )) || die "only ${free_gib} GiB free; require at least ${MIN_FREE_GIB} GiB"
  echo "Disk preflight : ${free_gib} GiB free"
fi

ORCH_ROOT="${OUT_ROOT}/.orchestration/runs/${RUN_ID}"
NODE_DIR="${ORCH_ROOT}/nodes"
T5_STATE_DIR="${ORCH_ROOT}/t5"
TASK_STATE_DIR="${ORCH_ROOT}/tasks"
MERGE_STATE_DIR="${ORCH_ROOT}/merge"
mkdir -p "${NODE_DIR}" "${T5_STATE_DIR}" "${TASK_STATE_DIR}" "${MERGE_STATE_DIR}"

write_registration() {
  python - "${NODE_DIR}/node_${NODE_RANK}.json" "${NODE_RANK}" "${NUM_NODES}" \
    "${PART_INDICES}" "${GPU_IDS}" "${RUN_ID}" "${OUT_ROOT}" "${PROMPTS_FILE}" \
    "${TRAIN_PROMPTS},${VAL_PROMPTS},${TEST_PROMPTS}" "${TRAIN_SEEDS}" "${EVAL_SEEDS}" \
    "${PRIMARY_LAMBDA}" "${LATENT_SAVE_DTYPE}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
prompts_file = Path(sys.argv[8]).resolve()
payload = {"schema": "oracle_1500_resume_node_v1", "node_rank": int(sys.argv[2]),
           "num_nodes": int(sys.argv[3]), "parts": [int(v) for v in sys.argv[4].split(",")],
           "gpus": [int(v) for v in sys.argv[5].split(",")], "run_id": sys.argv[6],
           "out_root": str(Path(sys.argv[7]).resolve()), "prompts_file": str(prompts_file),
           "prompts_sha256": hashlib.sha256(prompts_file.read_bytes()).hexdigest(),
           "split_counts": [int(v) for v in sys.argv[9].split(",")],
           "train_seeds": [int(v) for v in sys.argv[10].split()],
           "eval_seeds": [int(v) for v in sys.argv[11].split()],
           "primary_lambda": float(sys.argv[12]), "latent_dtype": sys.argv[13]}
if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != payload:
    raise SystemExit(f"Node registration changed for the same RUN_ID: {path}")
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

wait_for_path() {
  local path="$1" description="$2" started
  started="$(date +%s)"
  while [[ ! -s "${path}" ]]; do
    if (( WAIT_TIMEOUT > 0 && $(date +%s) - started >= WAIT_TIMEOUT )); then die "timed out waiting for ${description}: ${path}"; fi
    echo "[$(date '+%F %T')] Waiting for ${description}..."
    sleep "${WAIT_INTERVAL}"
  done
}

write_registration
for ((rank=0; rank<NUM_NODES; rank++)); do wait_for_path "${NODE_DIR}/node_${rank}.json" "node ${rank} registration"; done

python - "${NODE_DIR}" "${NUM_NODES}" <<'PY'
import json, sys
from pathlib import Path
root, num_nodes = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads((root / f"node_{rank}.json").read_text(encoding="utf-8")) for rank in range(num_nodes)]
parts = [part for row in rows for part in row["parts"]]
if sorted(parts) != list(range(8)) or len(parts) != len(set(parts)):
    raise SystemExit(f"Node registrations must cover canonical parts 0..7 exactly once; got {parts}")
if num_nodes == 2 and any(len(row["gpus"]) != 8 or len(row["parts"]) != 4 for row in rows):
    raise SystemExit("Two-node mode expects 8 GPUs and 4 parts on each node")
signature_keys = ("num_nodes", "out_root", "prompts_sha256", "split_counts", "train_seeds", "eval_seeds", "primary_lambda", "latent_dtype")
if any(any(row.get(key) != rows[0].get(key) for key in signature_keys) for row in rows[1:]):
    raise SystemExit("Node registrations disagree on dataset protocol or shared output root")
PY

PLAN_PATH="${OUT_ROOT}/generation_plan.json"
RESUME_PLAN="${ORCH_ROOT}/resume_plan.json"
if (( NODE_RANK == 0 )); then
  python - "${PLAN_PATH}" "${RESUME_PLAN}" "${NODE_DIR}" "${PROMPTS_FILE}" \
    "${BASE_PROMPT_OFFSET}" "${TRAIN_PROMPTS}" "${VAL_PROMPTS}" "${TEST_PROMPTS}" \
    "${TRAIN_SEEDS}" "${EVAL_SEEDS}" "${PRIMARY_LAMBDA}" "${EXPECTED_TOTAL_VIDEOS}" \
    "${EXPECTED_LATENT_FILES}" "${LATENT_SAVE_DTYPE}" <<'PY'
import datetime as dt
import json, os, sys
from pathlib import Path
(plan_path, resume_path, node_dir, prompts_file, base, train, val, test,
 train_seeds, eval_seeds, primary_lambda, expected_videos, expected_latents, latent_dtype) = sys.argv[1:]
base, train, val, test = map(int, (base, train, val, test))
steps = [30, 35, *range(40, 51)]
splits = {
    "train": {"prompt_offset": base, "prompt_count": train, "seeds": [int(v) for v in train_seeds.split()]},
    "validation": {"prompt_offset": base + train, "prompt_count": val, "seeds": [int(v) for v in eval_seeds.split()]},
    "test": {"prompt_offset": base + train + val, "prompt_count": test, "seeds": [int(v) for v in eval_seeds.split()]},
}
plan = Path(plan_path)
if plan.is_file():
    current = json.loads(plan.read_text(encoding="utf-8"))
    for name, spec in splits.items():
        for key in ("prompt_offset", "prompt_count", "seeds"):
            if current.get("splits", {}).get(name, {}).get(key) != spec[key]:
                raise SystemExit(f"Existing generation plan mismatch: {name}.{key}")
    if current.get("candidate_steps") != steps:
        raise SystemExit("Existing generation plan candidate steps mismatch")
    artifacts = current.get("artifacts", {})
    if artifacts.get("save_switch_latents") is not True or artifacts.get("preserve_candidate_videos") is not True:
        raise SystemExit("Existing generation plan does not preserve required videos/latents")
else:
    payload = {"schema": "oracle_1500_8gpu_generation_plan_v1",
               "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "prompts_file": str(Path(prompts_file).resolve()), "candidate_steps": steps,
               "primary_lambda": float(primary_lambda),
               "splits": {
                   "train": {**splits["train"], "include_native_hr": True, "physical_dataset": "train"},
                   "validation": {**splits["validation"], "include_native_hr": True, "physical_dataset": "eval"},
                   "test": {**splits["test"], "include_native_hr": True, "physical_dataset": "eval"}},
               "expected_videos": {"total": int(expected_videos)},
               "artifacts": {"preserve_candidate_videos": True, "save_switch_latents": True,
                   "latent_schema": "wan_taa_free_oracle_latent_v1", "latent_tensors": ["x_t_lr", "x0_pred_lr"],
                   "latent_dtype": latent_dtype, "expected_latent_files": int(expected_latents)}}
    temporary = plan.with_name(f".{plan.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, plan)
nodes = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(node_dir).glob("node_*.json"))]
resume = {"schema": "oracle_1500_two_node_resume_plan_v1",
          "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "generation_plan": str(plan.resolve()),
          "canonical_parts": 8, "lanes_per_part": 2, "resume_tasks": 16, "nodes": nodes}
resume_path = Path(resume_path)
temporary = resume_path.with_name(f".{resume_path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(resume, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, resume_path)
PY
fi
wait_for_path "${RESUME_PLAN}" "coordinator resume plan"

PROTOCOL_PREFLIGHT="${ORCH_ROOT}/existing_protocols.verified.json"
if (( NODE_RANK == 0 )); then
  python - "${OUT_ROOT}" "${PROMPTS_FILE}" "${BASE_PROMPT_OFFSET}" \
    "${TRAIN_PROMPTS}" "$((VAL_PROMPTS + TEST_PROMPTS))" "${LATENT_SAVE_DTYPE}" \
    "${PROTOCOL_PREFLIGHT}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

root, prompts_file = Path(sys.argv[1]), Path(sys.argv[2])
base, train_count, eval_count = map(int, sys.argv[3:6])
latent_dtype, marker = sys.argv[6], Path(sys.argv[7])
with prompts_file.open("r", encoding="utf-8") as handle:
    prompts = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
if len(prompts) < base + train_count + eval_count:
    raise SystemExit("Prompt file does not contain the configured 1,500-prompt slice")

def ranges(offset, count):
    base_count, remainder = divmod(count, 8)
    current = offset
    for part in range(8):
        size = base_count + int(part < remainder)
        yield part, current, size
        current += size

checked = []
steps = [30, 35, *range(40, 51)]
for split, offset, count, seeds in (("train", base, train_count, [42]), ("eval", base + train_count, eval_count, [42, 100, 2024])):
    for part, part_offset, part_count in ranges(offset, count):
        part_root = root / split / "_parts" / f"part_{part:02d}"
        selected = prompts[part_offset : part_offset + part_count]
        payload = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
        expected_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for seed in seeds:
            seed_root = part_root / "raw_samples" / f"seed_{seed}"
            protocol_path = seed_root / "protocol.json"
            has_raw_artifacts = any(path.is_file() for path in seed_root.glob("manifests/*.json"))
            if not protocol_path.is_file():
                if has_raw_artifacts:
                    raise SystemExit(f"Existing artifacts lack protocol.json: {seed_root}")
                continue
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            expected = {
                "prompt_offset": part_offset,
                "prompt_count": part_count,
                "selected_prompts_sha256": expected_hash,
                "candidate_steps": steps,
                "save_latents": True,
                "latent_save_dtype": latent_dtype,
                "include_native_hr": True,
                "start_seed": seed,
            }
            mismatches = {key: (protocol.get(key), value) for key, value in expected.items() if protocol.get(key) != value}
            if mismatches:
                raise SystemExit(f"Existing protocol mismatch at {protocol_path}: {mismatches}")
            checked.append(str(protocol_path.resolve()))
temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps({"verified": True, "checked_protocols": checked}, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker)
PY
fi
wait_for_path "${PROTOCOL_PREFLIGHT}" "existing protocol compatibility preflight"

verify_t5_slice() {
  local t5_dir="$1" offset="$2" count="$3"
  python - "${t5_dir}" "${offset}" "${count}" <<'PY'
import json, sys
from pathlib import Path
root, offset, count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for prompt_id in range(offset, offset + count):
    npz, meta = root / f"prompt_{prompt_id:06d}.npz", root / f"prompt_{prompt_id:06d}.json"
    if not npz.is_file() or npz.stat().st_size == 0 or not meta.is_file() or meta.stat().st_size == 0:
        raise SystemExit(3)
    if int(json.loads(meta.read_text(encoding="utf-8")).get("prompt_id", -1)) != prompt_id:
        raise SystemExit(3)
PY
}

prepare_t5() {
  local split_name="$1" split_offset="$2" split_count="$3"
  local t5_dir="${OUT_ROOT}/${split_name}/t5_embeddings" marker="${T5_STATE_DIR}/${split_name}.done.json"
  mkdir -p "${t5_dir}"
  if (( NODE_RANK == 0 )); then
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '{"dry_run":true}\n' > "${marker}"
    elif verify_t5_slice "${t5_dir}" "${split_offset}" "${split_count}"; then
      printf '{"reused":true}\n' > "${marker}"
    else
      [[ "${EXTRACT_T5}" == "1" ]] || die "T5 coverage incomplete and EXTRACT_T5=0"
      python "${SCRIPT_DIR}/extract_prompt_t5_embeddings.py" --prompts_file "${PROMPTS_FILE}" \
        --out_dir "${t5_dir}" --prompt_offset "${split_offset}" --limit "${split_count}" \
        --device "cuda:${GPU_ARRAY[0]}" --skip_existing
      verify_t5_slice "${t5_dir}" "${split_offset}" "${split_count}" || die "T5 verification failed for ${split_name}"
      printf '{"extracted":true}\n' > "${marker}"
    fi
  fi
  wait_for_path "${marker}" "${split_name} T5 completion"
  if [[ "${DRY_RUN}" != "1" ]]; then verify_t5_slice "${t5_dir}" "${split_offset}" "${split_count}" || die "shared T5 coverage invalid"; fi
}

task_verify() {
  local part_root="$1" offset="$2" count="$3" seeds="$4" include_native="$5" marker="$6"
  local args=(--part-root "${part_root}" --prompt-offset "${offset}" --limit "${count}" --seeds ${seeds}
    --candidate-steps "${CANDIDATE_STEPS[@]}" --include-native-hr "${include_native}"
    --require-latents 1 --marker "${marker}" --quiet)
  if [[ "${DRY_RUN}" == "1" ]]; then args+=(--dry-run); fi
  python "${SCRIPT_DIR}/verify_oracle_resume_task.py" "${args[@]}"
}

worker_pids=()
cleanup_workers() {
  local pid
  for pid in "${worker_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then pkill -TERM -P "${pid}" 2>/dev/null || true; kill -TERM "${pid}" 2>/dev/null || true; fi
  done
}

run_split() {
  local split_name="$1" split_offset="$2" split_count="$3" seeds="$4" include_native="$5"
  local split_root="${OUT_ROOT}/${split_name}" parts_dir="${OUT_ROOT}/${split_name}/_parts"
  local t5_dir="${OUT_ROOT}/${split_name}/t5_embeddings" split_task_dir="${TASK_STATE_DIR}/${split_name}"
  local split_log_dir="${OUT_ROOT}/logs/${split_name}/${RUN_ID}/${NODE_NAME}"
  local merge_marker="${MERGE_STATE_DIR}/${split_name}.done.json"
  mkdir -p "${parts_dir}" "${split_task_dir}" "${split_log_dir}"
  prepare_t5 "${split_name}" "${split_offset}" "${split_count}"

  local task_parts=() task_lanes=() task_offsets=() task_counts=()
  local part lane canonical_offset canonical_count lane0_count lane_offset lane_count
  for part in "${PART_ARRAY[@]}"; do
    read -r canonical_offset canonical_count < <(split_part_range "${split_offset}" "${split_count}" "${part}")
    lane0_count=$(((canonical_count + 1) / 2))
    for lane in 0 1; do
      if (( lane == 0 )); then lane_offset="${canonical_offset}"; lane_count="${lane0_count}";
      else lane_offset=$((canonical_offset + lane0_count)); lane_count=$((canonical_count - lane0_count)); fi
      (( lane_count > 0 )) || continue
      task_parts+=("${part}"); task_lanes+=("${lane}"); task_offsets+=("${lane_offset}"); task_counts+=("${lane_count}")
    done
  done

  echo "[${split_name}] ${#task_parts[@]} resume tasks on ${#GPU_ARRAY[@]} GPUs"
  worker_pids=()
  local worker_idx gpu worker_log
  for worker_idx in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$worker_idx]}"
    worker_log="${split_log_dir}/worker_${worker_idx}_gpu${gpu}.log"
    (
      local task_idx p l o c p_name p_root marker task_log canonical_o canonical_c
      for ((task_idx=worker_idx; task_idx<${#task_parts[@]}; task_idx+=${#GPU_ARRAY[@]})); do
        p="${task_parts[$task_idx]}"; l="${task_lanes[$task_idx]}"; o="${task_offsets[$task_idx]}"; c="${task_counts[$task_idx]}"
        p_name="$(printf 'part_%02d' "${p}")"; p_root="${parts_dir}/${p_name}"
        marker="${split_task_dir}/${p_name}_lane${l}.done.json"; task_log="${split_log_dir}/${p_name}_lane${l}.log"
        read -r canonical_o canonical_c < <(split_part_range "${split_offset}" "${split_count}" "${p}")
        if task_verify "${p_root}" "${o}" "${c}" "${seeds}" "${include_native}" "${marker}"; then
          echo "[GPU ${gpu}] Reusing complete ${p_name}/lane${l} offset=${o} count=${c}"
          continue
        fi
        echo "[GPU ${gpu}] Resuming ${p_name}/lane${l} offset=${o} count=${c}"
        GPU_ID="${gpu}" PROMPT_OFFSET="${o}" LIMIT="${c}" PROTOCOL_PROMPT_OFFSET="${canonical_o}" \
          PROTOCOL_PROMPT_LIMIT="${canonical_c}" SEEDS="${seeds}" PROMPTS_FILE="${PROMPTS_FILE}" \
          OUT_ROOT="${p_root}" T5_EMBED_DIR="${t5_dir}" SKIP_EXISTING=1 DRY_RUN="${DRY_RUN}" \
          CLEAN_VIDEOS=0 INCLUDE_NATIVE_HR="${include_native}" SAVE_LATENTS=1 \
          ENABLE_INLINE_VBENCH=0 \
          LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE}" PRIMARY_LAMBDA="${PRIMARY_LAMBDA}" \
          bash "${SCRIPT_DIR}/run_oracle_worker.sh" >"${task_log}" 2>&1
        task_verify "${p_root}" "${o}" "${c}" "${seeds}" "${include_native}" "${marker}" || {
          echo "Task verification failed: ${p_name}/lane${l}; log=${task_log}" >&2; exit 1; }
        echo "[GPU ${gpu}] Completed ${p_name}/lane${l}"
      done
    ) >"${worker_log}" 2>&1 &
    worker_pids+=("$!")
  done

  trap 'cleanup_workers; exit 130' INT TERM
  local remaining="${#worker_pids[@]}" failed=0 pid i
  local finished=()
  for _ in "${worker_pids[@]}"; do finished+=(0); done
  while (( remaining > 0 )); do
    sleep "${MONITOR_INTERVAL}"
    for i in "${!worker_pids[@]}"; do
      (( finished[i] == 0 )) || continue
      pid="${worker_pids[$i]}"
      if ! kill -0 "${pid}" 2>/dev/null; then
        if ! wait "${pid}"; then failed=1; fi
        finished[i]=1; remaining=$((remaining - 1))
      fi
    done
    echo "[$(date '+%F %T')] [${split_name}] node ${NODE_RANK}: ${remaining} workers running"
    if (( failed != 0 )); then cleanup_workers; wait || true; trap - INT TERM; die "worker failure; inspect ${split_log_dir}"; fi
  done
  trap - INT TERM

  local expected_marker
  for ((part=0; part<NUM_CANONICAL_PARTS; part++)); do
    for lane in 0 1; do
      expected_marker="${split_task_dir}/$(printf 'part_%02d' "${part}")_lane${lane}.done.json"
      wait_for_path "${expected_marker}" "${split_name} part ${part} lane ${lane}"
    done
  done

  if (( NODE_RANK == 0 )); then
    echo "[${split_name}] All 16 tasks complete; verifying before coordinator merge..."
    local o c p_name marker
    for ((part=0; part<NUM_CANONICAL_PARTS; part++)); do
      read -r canonical_offset canonical_count < <(split_part_range "${split_offset}" "${split_count}" "${part}")
      lane0_count=$(((canonical_count + 1) / 2))
      for lane in 0 1; do
        if (( lane == 0 )); then o="${canonical_offset}"; c="${lane0_count}";
        else o=$((canonical_offset + lane0_count)); c=$((canonical_count - lane0_count)); fi
        p_name="$(printf 'part_%02d' "${part}")"; marker="${split_task_dir}/${p_name}_lane${lane}.done.json"
        task_verify "${parts_dir}/${p_name}" "${o}" "${c}" "${seeds}" "${include_native}" "${marker}" || die "final task verification failed: ${split_name}/${p_name}/lane${lane}"
      done
    done
    python "${SCRIPT_DIR}/merge_and_verify_oracle_dataset.py" --parts_dir "${parts_dir}" \
      --out_root "${split_root}" --total_prompts "${split_count}" --seeds ${seeds} --primary_lambda "${PRIMARY_LAMBDA}"
    python - "${split_root}/dataset_manifest.json" "${merge_marker}" <<'PY'
import json, os, sys
from pathlib import Path
manifest, marker = Path(sys.argv[1]), Path(sys.argv[2])
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("is_complete") is not True: raise SystemExit("merged dataset manifest is incomplete")
temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps({"complete": True, "manifest": str(manifest.resolve())}, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker)
PY
  fi
  wait_for_path "${merge_marker}" "${split_name} coordinator merge"
}

run_split "train" "${TRAIN_OFFSET}" "${TRAIN_PROMPTS}" "${TRAIN_SEEDS}" 1
run_split "eval" "${EVAL_OFFSET}" "$((VAL_PROMPTS + TEST_PROMPTS))" "${EVAL_SEEDS}" 1

COMPLETION_PATH="${OUT_ROOT}/generation_complete.json"
RUN_COMPLETION_PATH="${ORCH_ROOT}/generation_complete.done.json"
if (( NODE_RANK == 0 )); then
  python - "${COMPLETION_PATH}" "${PLAN_PATH}" "${OUT_ROOT}/train/dataset_manifest.json" \
    "${OUT_ROOT}/eval/dataset_manifest.json" "${OUT_ROOT}/train" "${OUT_ROOT}/eval" "${DRY_RUN}" <<'PY'
import datetime as dt
import json, os, sys
from pathlib import Path
output, plan_path, train_path, eval_path, train_root, eval_root = map(Path, sys.argv[1:7])
dry_run = bool(int(sys.argv[7])); plan = json.loads(plan_path.read_text(encoding="utf-8")); manifests = {}
for name, path in (("train", train_path), ("eval", eval_path)):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("is_complete") is not True: raise SystemExit(f"{name} dataset is incomplete: {path}")
    manifests[name] = {"path": str(path), "is_complete": True}
candidate_count = len(plan["candidate_steps"]); train_spec = plan["splits"]["train"]
val_spec, test_spec = plan["splits"]["validation"], plan["splits"]["test"]
specs = (("train", train_root, train_spec["prompt_count"] * len(train_spec["seeds"])),
         ("eval", eval_root, (val_spec["prompt_count"] + test_spec["prompt_count"]) * len(val_spec["seeds"])))
coverage = {}
for name, root, trajectories in specs:
    def count(pattern): return sum(1 for p in root.rglob(pattern) if p.is_file() and p.stat().st_size > 0)
    actual = {"candidate_videos": count("videos/step*/*.mp4"), "native_videos": count("videos/native_hr/*.mp4"),
              "switch_latents": count("latents/step*/*.pt")}
    expected = {"candidate_videos": 0 if dry_run else trajectories * candidate_count,
                "native_videos": 0 if dry_run else trajectories,
                "switch_latents": 0 if dry_run else trajectories * candidate_count}
    if actual != expected: raise SystemExit(f"{name} artifact coverage mismatch: actual={actual}, expected={expected}")
    coverage[name] = {"actual": actual, "expected": expected, "verified": True}
result = {"schema": "oracle_1500_8gpu_generation_complete_v1",
          "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "plan": str(plan_path),
          "manifests": manifests, "artifact_coverage": coverage, "resumed_with_two_nodes_16_gpus": True}
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, output)
PY
  python - "${COMPLETION_PATH}" "${RUN_COMPLETION_PATH}" <<'PY'
import json, os, sys
from pathlib import Path
completion, marker = Path(sys.argv[1]), Path(sys.argv[2])
payload = json.loads(completion.read_text(encoding="utf-8"))
temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps({"complete": True, "completion": str(completion.resolve()), "schema": payload.get("schema")}, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker)
PY
fi
wait_for_path "${RUN_COMPLETION_PATH}" "final generation completion"
echo "================================================================================"
echo " Resume complete and verified: ${COMPLETION_PATH}"
echo "================================================================================"
