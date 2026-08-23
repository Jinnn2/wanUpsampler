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

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
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
if (( TRAIN_PROMPTS < 1 || VAL_PROMPTS < 1 || TEST_PROMPTS < 1 )); then
  echo "TRAIN_PROMPTS, VAL_PROMPTS, and TEST_PROMPTS must all be positive." >&2
  exit 2
fi

TOTAL_PROMPTS=$((TRAIN_PROMPTS + VAL_PROMPTS + TEST_PROMPTS))
if (( TOTAL_PROMPTS != 1500 )); then
  echo "This formal launcher requires exactly 1500 prompts; configured total=${TOTAL_PROMPTS}." >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "GPU_IDS must contain exactly 8 devices, got ${#GPU_ARRAY[@]}: ${GPU_IDS}" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ -z "${gpu}" || -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU_IDS must contain 8 unique, non-empty device ids: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu}"]=1
done

read -r -a TRAIN_SEED_ARRAY <<< "${TRAIN_SEEDS}"
read -r -a EVAL_SEED_ARRAY <<< "${EVAL_SEEDS}"
if (( ${#TRAIN_SEED_ARRAY[@]} != 1 )); then
  echo "TRAIN_SEEDS must contain exactly one seed for the 1500-prompt efficient design." >&2
  exit 2
fi
if (( ${#EVAL_SEED_ARRAY[@]} != 3 )); then
  echo "EVAL_SEEDS must contain exactly three seeds for formal validation/test." >&2
  exit 2
fi
if [[ "${TRAIN_INCLUDE_NATIVE_HR}" != "0" && "${TRAIN_INCLUDE_NATIVE_HR}" != "1" ]]; then
  echo "TRAIN_INCLUDE_NATIVE_HR must be 0 or 1, got: ${TRAIN_INCLUDE_NATIVE_HR}" >&2
  exit 2
fi
if [[ "${CLEAN_VIDEOS}" != "0" ]]; then
  echo "CLEAN_VIDEOS must remain 0: this launcher preserves every candidate video." >&2
  exit 2
fi
if [[ "${SAVE_LATENTS}" != "1" ]]; then
  echo "SAVE_LATENTS must remain 1: this launcher preserves every switch-step latent." >&2
  exit 2
fi
if [[ "${LATENT_SAVE_DTYPE}" != "fp16" && "${LATENT_SAVE_DTYPE}" != "bf16" && "${LATENT_SAVE_DTYPE}" != "fp32" ]]; then
  echo "LATENT_SAVE_DTYPE must be fp16, bf16, or fp32, got: ${LATENT_SAVE_DTYPE}" >&2
  exit 2
fi

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
# Each archive contains x_t_lr and x0_pred_lr with shape [1,16,21,46,80].
ESTIMATED_LATENT_BYTES=$((EXPECTED_LATENT_FILES * 2 * 16 * 21 * 46 * 80 * LATENT_BYTES_PER_VALUE))
ESTIMATED_LATENT_GIB=$(((ESTIMATED_LATENT_BYTES + 1073741823) / 1073741824))

echo "================================================================================"
echo " 1500-Prompt / 8-GPU Oracle Video Generation Plan"
echo "================================================================================"
echo " GPUs       : ${GPU_IDS}"
echo " Prompt file: ${PROMPTS_FILE}"
echo " Output root: ${OUT_ROOT}"
echo " Train      : prompts [${TRAIN_OFFSET}, $((EVAL_OFFSET - 1))], count=${TRAIN_PROMPTS}, seeds='${TRAIN_SEEDS}', native=${TRAIN_INCLUDE_NATIVE_HR}"
echo " Validation : prompts [${VAL_OFFSET}, $((TEST_OFFSET - 1))], count=${VAL_PROMPTS}, seeds='${EVAL_SEEDS}', native=1"
echo " Test       : prompts [${TEST_OFFSET}, $((END_OFFSET - 1))], count=${TEST_PROMPTS}, seeds='${EVAL_SEEDS}', native=1"
echo " Physical datasets: train=${OUT_ROOT}/train, eval=${OUT_ROOT}/eval"
echo " Expected videos : train=${EXPECTED_TRAIN_VIDEOS}, eval=${EXPECTED_EVAL_VIDEOS}, total=${EXPECTED_TOTAL_VIDEOS}"
echo " Switch latents  : ${EXPECTED_LATENT_FILES} files, ${LATENT_SAVE_DTYPE}, approximately ${ESTIMATED_LATENT_GIB} GiB"
echo " Preserve videos : yes (CLEAN_VIDEOS=0)"
echo " Resume/skip     : ${SKIP_EXISTING}"
echo " Dry run         : ${DRY_RUN}"
echo "================================================================================"

if [[ "${TRAIN_INCLUDE_NATIVE_HR}" == "0" ]]; then
  echo "WARNING: train/native_hr is disabled. The current strict VBench scorer requires" >&2
  echo "Native-HR per trajectory, so train will be generation-only until a calibrated" >&2
  echo "native-latency training profile is used. Validation/test remain strict-ready." >&2
fi

if [[ "${PLAN_ONLY}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUT_ROOT}"
PLAN_PATH="${OUT_ROOT}/generation_plan.json"
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

run_dataset() {
  local split_name="$1"
  local prompt_offset="$2"
  local prompt_count="$3"
  local seeds="$4"
  local include_native="$5"
  local split_root="${OUT_ROOT}/${split_name}"
  local split_logs="${OUT_ROOT}/logs/${split_name}"

  echo "Starting physical split '${split_name}' on all 8 GPUs..."
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    GPU_IDS="${GPU_IDS}" \
    TOTAL_PROMPTS="${prompt_count}" \
    EXPECTED_TOTAL_PROMPTS="${prompt_count}" \
    PROMPT_OFFSET="${prompt_offset}" \
    PROMPTS_FILE="${PROMPTS_FILE}" \
    SEEDS="${seeds}" \
    OUT_ROOT="${split_root}" \
    LOG_DIR="${split_logs}" \
    EXTRACT_T5="${EXTRACT_T5}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    CLEAN_VIDEOS="${CLEAN_VIDEOS}" \
    INCLUDE_NATIVE_HR="${include_native}" \
    SAVE_LATENTS="${SAVE_LATENTS}" \
    LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE}" \
    DRY_RUN="${DRY_RUN}" \
    PRIMARY_LAMBDA="${PRIMARY_LAMBDA}" \
    MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
    MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES}" \
    bash "${SCRIPT_DIR}/build_oracle_dataset_4gpu.sh"
}

# Keep all eight devices occupied and minimize model reloads: validation and test
# share one physical 500-prompt dataset and are separated by prompt-id ranges.
run_dataset "train" "${TRAIN_OFFSET}" "${TRAIN_PROMPTS}" "${TRAIN_SEEDS}" "${TRAIN_INCLUDE_NATIVE_HR}"
run_dataset "eval" "${EVAL_OFFSET}" "$((VAL_PROMPTS + TEST_PROMPTS))" "${EVAL_SEEDS}" "1"

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

echo "1500-prompt generation completed."
echo "Plan     : ${PLAN_PATH}"
echo "Completion: ${OUT_ROOT}/generation_complete.json"
