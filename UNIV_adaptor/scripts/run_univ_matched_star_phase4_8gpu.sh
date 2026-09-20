#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  check|prepare|plan|generate|finalize|all) ;;
  *) echo "Usage: $0 [check|prepare|plan|generate|finalize|all]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
REALESRGAN_REPO="${REALESRGAN_REPO:-/mnt/afs_2/houze/Real-ESRGAN}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
PROTOCOL="${PROTOCOL:-${PROJECT_ROOT}/UNIV_adaptor/configs/univ_matched_star_phase4.json}"
SOURCE_PHASE3_ROOT="${SOURCE_PHASE3_ROOT:-${PROJECT_ROOT}/outputs/univ_sparse_action_phase3_v1}"
NEW_PROMPTS_FILE="${NEW_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/univ_controller_pilot_500.txt}"
NEW_PROMPT_OFFSET="${NEW_PROMPT_OFFSET:-181}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${PROJECT_ROOT}/UNIV_adaptor/configs/wan21_t2v_univ_rgb_720p.example.json}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_matched_star_phase4_v1}"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/run_matched_star_generation.py"
MANIFEST="${OUT_ROOT}/generation_manifest.json"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
JOB_CHUNK_SIZE="${JOB_CHUNK_SIZE:-32}"
MAX_JOBS_PER_WORKER="${MAX_JOBS_PER_WORKER:-0}"
RESUME="${RESUME:-1}"
EXPECTED_GENERATED="${EXPECTED_GENERATED:-990}"
EXPECTED_REUSED="${EXPECTED_REUSED:-366}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-camera shake, overexposed, static image, blurry details, subtitles, text, watermark, low quality, jpeg artifacts, distorted hands, distorted face, malformed body, duplicate limbs}"
export DTYPE="${DTYPE:-BF16}"
GENERATION_LOCK_DIR=""

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight comma-separated ids." >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ || -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU_IDS must contain eight unique non-negative integers: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu}"]=1
done
[[ "${JOB_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "JOB_CHUNK_SIZE must be positive." >&2; exit 2; }
[[ "${NEW_PROMPT_OFFSET}" =~ ^[0-9]+$ ]] || { echo "NEW_PROMPT_OFFSET must be non-negative." >&2; exit 2; }
[[ "${MAX_JOBS_PER_WORKER}" =~ ^[0-9]+$ ]] || { echo "MAX_JOBS_PER_WORKER must be non-negative." >&2; exit 2; }
[[ "${RESUME}" == "0" || "${RESUME}" == "1" ]] || { echo "RESUME must be 0 or 1." >&2; exit 2; }
[[ "${EXPECTED_GENERATED}" =~ ^[0-9]+$ && "${EXPECTED_REUSED}" =~ ^[0-9]+$ ]] || {
  echo "EXPECTED_GENERATED and EXPECTED_REUSED must be non-negative integers." >&2
  exit 2
}

require_driver() {
  [[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
  [[ -f "${DRIVER}" ]] || { echo "Required file not found: ${DRIVER}" >&2; exit 1; }
}

require_prepare_inputs() {
  require_driver
  for path in "${PROTOCOL}" "${NEW_PROMPTS_FILE}" "${TEMPLATE_CONFIG}"; do
    [[ -f "${path}" ]] || { echo "Required file not found: ${path}" >&2; exit 1; }
  done
  for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}" "${SOURCE_PHASE3_ROOT}"; do
    [[ -d "${path}" ]] || { echo "Required directory not found: ${path}" >&2; exit 1; }
  done
  [[ -f "${SOURCE_PHASE3_ROOT}/sparse_dataset_manifest.json" ]] || {
    echo "Phase3 dataset manifest not found: ${SOURCE_PHASE3_ROOT}/sparse_dataset_manifest.json" >&2
    exit 1
  }
}

require_generation_inputs() {
  require_driver
  for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
    [[ -d "${path}" ]] || { echo "Required directory not found: ${path}" >&2; exit 1; }
  done
}

validate_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required." >&2; exit 1; }
  mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  declare -A AVAILABLE_SET=()
  for gpu in "${AVAILABLE_GPUS[@]}"; do AVAILABLE_SET["${gpu//[[:space:]]/}"]=1; done
  for gpu in "${GPU_ARRAY[@]}"; do
    [[ -n "${AVAILABLE_SET[${gpu}]:-}" ]] || { echo "GPU ${gpu} is unavailable." >&2; exit 1; }
  done
}

prepare_manifest() {
  require_prepare_inputs
  "${WAN_PYTHON}" "${DRIVER}" prepare \
    --protocol "${PROTOCOL}" \
    --source-phase3-root "${SOURCE_PHASE3_ROOT}" \
    --new-prompts "${NEW_PROMPTS_FILE}" \
    --new-prompt-offset "${NEW_PROMPT_OFFSET}" \
    --template-config "${TEMPLATE_CONFIG}" \
    --model-root "${MODEL_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --job-chunk-size "${JOB_CHUNK_SIZE}" \
    --worker-count 8
}

list_worker_jobs() {
  local slot="$1"
  local -a args=("${DRIVER}" list-jobs --manifest "${MANIFEST}" --worker-slot "${slot}")
  (( MAX_JOBS_PER_WORKER > 0 )) && args+=(--limit "${MAX_JOBS_PER_WORKER}")
  "${WAN_PYTHON}" "${args[@]}"
}

print_plan() {
  require_driver
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  "${WAN_PYTHON}" - "${MANIFEST}" "${OUT_ROOT}" "${EXPECTED_GENERATED}" "${EXPECTED_REUSED}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
plan = json.loads(Path(manifest["plan_path"]).read_text(encoding="utf-8"))
print("UNIV Phase4 matched-star frozen plan")
print(f"  source Phase3: {plan['source_phase3']['root']}")
print(f"  new prompts: {plan['new_train_prompts_file']}")
print(f"  new prompt offset/count: {plan['new_train_prompt_offset']}/{plan['new_train_prompt_count']}")
print(f"  output: {Path(sys.argv[2]).resolve()}")
print(f"  counts: {json.dumps(plan['counts'], sort_keys=True)}")
expected_generated = int(sys.argv[3])
expected_reused = int(sys.argv[4])
if plan["counts"]["generated_videos"] != expected_generated or plan["counts"]["reused_videos"] != expected_reused:
    raise SystemExit(
        "Phase4 plan count mismatch: "
        f"generated={plan['counts']['generated_videos']} expected={expected_generated}; "
        f"reused={plan['counts']['reused_videos']} expected={expected_reused}"
    )
PY
  for slot in 0 1 2 3 4 5 6 7; do
    local count
    count="$(list_worker_jobs "${slot}" | wc -l)"
    echo "  GPU ${GPU_ARRAY[${slot}]}: ${count} jobs"
  done
}

generate_parallel() {
  require_generation_inputs
  validate_gpus
  [[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }
  mkdir -p "${OUT_ROOT}/logs/8gpu_matched_star"
  GENERATION_LOCK_DIR="${OUT_ROOT}/.matched_star_generation.lock"
  if ! mkdir "${GENERATION_LOCK_DIR}" 2>/dev/null; then
    echo "Generation lock exists: ${GENERATION_LOCK_DIR}" >&2
    exit 1
  fi
  local -a pids=()
  cleanup() {
    if [[ -n "${GENERATION_LOCK_DIR:-}" ]]; then rmdir "${GENERATION_LOCK_DIR}" 2>/dev/null || true; fi
  }
  stop_children() {
    trap - INT TERM
    for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    cleanup
    exit 130
  }
  trap stop_children INT TERM
  trap cleanup EXIT
  for slot in 0 1 2 3 4 5 6 7; do
    local gpu="${GPU_ARRAY[${slot}]}"
    local log="${OUT_ROOT}/logs/8gpu_matched_star/gpu_${gpu}.log"
    (
      while IFS= read -r job_id; do
        [[ -n "${job_id}" ]] || continue
        args=(
          "${DRIVER}" generate-job
          --manifest "${MANIFEST}"
          --job-id "${job_id}"
          --wan-python "${WAN_PYTHON}"
          --lightx2v-repo "${LIGHTX2V_REPO}"
          --realesrgan-repo "${REALESRGAN_REPO}"
          --negative-prompt "${NEGATIVE_PROMPT}"
        )
        [[ "${RESUME}" == "1" ]] && args+=(--resume)
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
          "${WAN_PYTHON}" "${args[@]}"
      done < <(list_worker_jobs "${slot}")
    ) >>"${log}" 2>&1 &
    pids+=("$!")
    echo "[launch] GPU ${gpu}, worker ${slot} -> ${log}"
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  cleanup
  GENERATION_LOCK_DIR=""
  trap - EXIT INT TERM
  (( failed == 0 )) || { echo "At least one worker failed; inspect logs." >&2; exit 1; }
}

finalize_records() {
  "${WAN_PYTHON}" "${DRIVER}" finalize --manifest "${MANIFEST}" --out-root "${OUT_ROOT}"
}

case "${MODE}" in
  check)
    require_prepare_inputs
    validate_gpus
    "${WAN_PYTHON}" - "${PROTOCOL}" "${NEW_PROMPTS_FILE}" "${NEW_PROMPT_OFFSET}" <<'PY'
import json
import sys
from pathlib import Path
from UNIV_adaptor.sparse_action_protocol import action_catalog, validate_sparse_protocol

protocol = validate_sparse_protocol(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
catalog = action_catalog(protocol)
expected = {
    "REFERENCE": (0.75, 0.80, 0.55, 0.80),
    "STAR_S": (0.625, 0.80, 0.55, 0.80),
    "STAR_T": (0.75, 0.67, 0.55, 0.80),
    "STAR_C": (0.75, 0.80, 0.40, 0.80),
}
fields = ("spatial_ratio", "temporal_ratio", "lr_nfe_ratio", "switch_ratio")
for name, values in expected.items():
    observed = tuple(catalog[name]["requested_action"][field] for field in fields)
    if observed != values:
        raise SystemExit(f"matched-star action mismatch: {name}: {observed}")
prompts = [line.strip() for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
offset = int(sys.argv[3])
count = int(protocol["new_train_prompt_count"])
if len(prompts[offset:offset + count]) != count:
    raise SystemExit("new prompt slice is incomplete")
print("Matched-star protocol and 24-prompt slice passed")
PY
    PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${WAN_PYTHON}" - "${MODEL_ROOT}" <<'PY'
import sys
import torch
from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root
from lightx2v.common.ops import *  # noqa: F403
import UNIV_adaptor.wan_runner  # noqa: F401
from lightx2v.utils.registry_factory import RUNNER_REGISTER

config = validate_wan21_t2v_model_root(sys.argv[1])
if "wan2.1_univ_pipeline" not in RUNNER_REGISTER:
    raise SystemExit("runner registration missing: wan2.1_univ_pipeline")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"Wan model contract: dim={config['dim']}, heads={config['num_heads']}")
print(f"Runtime imports passed; visible CUDA devices={torch.cuda.device_count()}")
PY
    ;;
  prepare) prepare_manifest ;;
  plan) prepare_manifest; print_plan ;;
  generate) print_plan; generate_parallel ;;
  finalize) finalize_records ;;
  all) prepare_manifest; print_plan; generate_parallel; finalize_records ;;
esac
