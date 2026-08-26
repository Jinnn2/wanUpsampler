#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MODE="${1:-check}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
SCORED_ROOT="${SCORED_ROOT:-${GENERATION_ROOT}/scored_vbench5}"
STATE_DIR="${STATE_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection}"
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
  args=(
    --generation-root "${GENERATION_ROOT}"
    --scored-train-dir "${SCORED_ROOT}/train"
    --scored-eval-dir "${SCORED_ROOT}/eval"
    --output-dir "${STATE_DIR}"
    --splits train validation
    --progress-every "${FEATURE_PROGRESS_EVERY}"
    --torch-threads "${TORCH_THREADS}"
  )
  if [[ -d "${STATE_DIR}" ]]; then
    args+=(--skip-existing)
  fi
  python "${SCRIPT_DIR}/prepare_1500_variable_lambda_states.py" "${args[@]}"
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
  features)
    check_generation
    prepare_features
    ;;
  all)
    check_generation
    score_split train 1000 42
    score_split eval 500 42 100 2024
    prepare_features
    ;;
  *)
    echo "Usage: bash $0 {check|score|features|all}" >&2
    exit 2
    ;;
esac

echo "Scored datasets : ${SCORED_ROOT}/{train,eval}"
echo "Selection states: ${STATE_DIR}"
