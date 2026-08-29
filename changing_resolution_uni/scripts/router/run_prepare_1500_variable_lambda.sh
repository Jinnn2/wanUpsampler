#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MODE="${1:-check}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
SCORED_ROOT="${SCORED_ROOT:-${GENERATION_ROOT}/scored_vbench5}"
STATE_DIR="${STATE_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection}"
LATENCY_PROFILE="${LATENCY_PROFILE:-${SCORED_ROOT}/train_latency_profile_h100.json}"
HARDWARE_LABEL="${HARDWARE_LABEL:-H100}"
PROFILE_BOOTSTRAP_SAMPLES="${PROFILE_BOOTSTRAP_SAMPLES:-10000}"
PROFILE_BOOTSTRAP_SEED="${PROFILE_BOOTSTRAP_SEED:-2027}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-python}"
NGPUS="${NGPUS:-8}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.08}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-}"
FEATURE_PROGRESS_EVERY="${FEATURE_PROGRESS_EVERY:-10}"
TORCH_THREADS="${TORCH_THREADS:-4}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

check_generation() {
  python - "${GENERATION_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
plan_path = root / "generation_plan.json"
if not plan_path.is_file():
    raise SystemExit(f"Missing generation plan: {plan_path}")
plan = json.loads(plan_path.read_text(encoding="utf-8"))
print(json.dumps({
    "generation_root": str(root),
    "plan_schema": plan.get("schema"),
    "splits": plan.get("splits"),
    "expected_videos": plan.get("expected_videos"),
    "expected_latents": plan.get("artifacts", {}).get("expected_latent_files"),
}, indent=2, ensure_ascii=False))
completion = root / "generation_complete.json"
if not completion.is_file():
    raise SystemExit(
        "Generation is still incomplete: generation_complete.json is absent. "
        "Run this check again after the 8-GPU generator finishes."
    )
payload = json.loads(completion.read_text(encoding="utf-8"))
for split, coverage in payload.get("artifact_coverage", {}).items():
    if coverage.get("verified") is not True:
        raise SystemExit(f"Unverified artifact coverage for {split}: {coverage}")
print(f"Generation coverage verified: {completion}")
PY
}

score_split() {
  local physical_split="$1"
  local expected_prompts="$2"
  shift 2
  local expected_seeds=("$@")
  local scored_dir="${SCORED_ROOT}/${physical_split}"
  mkdir -p "${scored_dir}"
  args=(
    --input_dirs "${GENERATION_ROOT}/${physical_split}"
    --dataset_dir "${scored_dir}"
    --vbench_root "${VBENCH_ROOT}"
    --python "${VBENCH_PYTHON}"
    --ngpus "${NGPUS}"
    --expected_prompts "${expected_prompts}"
    --expected_seeds "${expected_seeds[@]}"
    --seed_policy prompt_offset
    --primary_lambda "${PRIMARY_LAMBDA}"
  )
  if [[ -n "${EXPECTED_VBENCH_COMMIT}" ]]; then
    args+=(--expected_vbench_commit "${EXPECTED_VBENCH_COMMIT}")
  fi
  python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/batch_vbench_score_dataset.py" "${args[@]}"
}

prepare_features() {
  if [[ ! -f "${LATENCY_PROFILE}" ]]; then
    echo "Missing locked train latency profile: ${LATENCY_PROFILE}" >&2
    echo "Run: bash $0 profile" >&2
    exit 2
  fi
  args=(
    --generation-root "${GENERATION_ROOT}"
    --scored-train-dir "${SCORED_ROOT}/train"
    --scored-eval-dir "${SCORED_ROOT}/eval"
    --output-dir "${STATE_DIR}"
    --latency-profile "${LATENCY_PROFILE}"
    --splits train validation
    --progress-every "${FEATURE_PROGRESS_EVERY}"
    --torch-threads "${TORCH_THREADS}"
  )
  if [[ -d "${STATE_DIR}" ]]; then
    args+=(--skip-existing)
  fi
  python "${SCRIPT_DIR}/prepare_1500_variable_lambda_states.py" "${args[@]}"
}

build_latency_profile() {
  python "${SCRIPT_DIR}/build_train_latency_profile.py" \
    --scored-train-dir "${SCORED_ROOT}/train" \
    --output "${LATENCY_PROFILE}" \
    --hardware-label "${HARDWARE_LABEL}" \
    --expected-prompts 1000 \
    --expected-base-seed 42 \
    --bootstrap-samples "${PROFILE_BOOTSTRAP_SAMPLES}" \
    --bootstrap-seed "${PROFILE_BOOTSTRAP_SEED}"
}

case "${MODE}" in
  check)
    check_generation
    ;;
  score)
    check_generation
    score_split train 1000 42
    score_split eval 500 42 100 2024
    ;;
  profile)
    check_generation
    build_latency_profile
    ;;
  features)
    check_generation
    prepare_features
    ;;
  all)
    check_generation
    score_split train 1000 42
    score_split eval 500 42 100 2024
    build_latency_profile
    prepare_features
    ;;
  *)
    echo "Usage: bash $0 {check|score|profile|features|all}" >&2
    exit 2
    ;;
esac

echo "Scored datasets : ${SCORED_ROOT}/{train,eval}"
echo "Selection states: ${STATE_DIR}"
echo "Latency profile : ${LATENCY_PROFILE}"
