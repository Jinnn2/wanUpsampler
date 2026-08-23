#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
SOURCE_DATASET_DIRS="${SOURCE_DATASET_DIRS:-${DATASET_DIR}}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
NGPUS="${NGPUS:-4}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-1000}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-42 100 2024}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
DIAGNOSTIC_DIMENSIONS="${DIAGNOSTIC_DIMENSIONS:-}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
read -r -a source_dataset_array <<< "${SOURCE_DATASET_DIRS}"

if ! python -c "import clip, pyiqa; assert hasattr(clip, 'load')" 2>/dev/null; then
  bash "${SCRIPT_DIR}/setup_environment.sh"
fi

score_args=(
  --input_dirs "${source_dataset_array[@]}"
  --dataset_dir "${DATASET_DIR}"
  --vbench_root "${VBENCH_ROOT}"
  --ngpus "${NGPUS}"
  --expected_prompts "${EXPECTED_PROMPTS}"
  --expected_seeds ${EXPECTED_SEEDS}
  --primary_lambda "${PRIMARY_LAMBDA}"
)
if [[ -n "${DIAGNOSTIC_DIMENSIONS}" ]]; then
  read -r -a diagnostic_dimension_array <<< "${DIAGNOSTIC_DIMENSIONS}"
  score_args+=(--diagnostic_dimensions "${diagnostic_dimension_array[@]}")
fi
if [[ "${FORCE_RESCORE}" == "1" ]]; then
  score_args+=(--force_rescore)
fi
if [[ -n "${EXPECTED_VBENCH_COMMIT}" ]]; then
  score_args+=(--expected_vbench_commit "${EXPECTED_VBENCH_COMMIT}")
fi

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/batch_vbench_score_dataset.py" \
  "${score_args[@]}"

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/cleanup_legacy_records.py" \
  --dataset_dir "${DATASET_DIR}" \
  --profile formal \
  --strict

echo "Strict VBench rescore and formal audit complete: ${DATASET_DIR}"
