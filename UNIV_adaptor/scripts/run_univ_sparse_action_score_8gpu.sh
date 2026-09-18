#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  check|score|report|all) ;;
  *) echo "Usage: $0 [check|score|report|all]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/outputs/univ_sparse_action_phase3_v1}"
OUT_DIR="${OUT_DIR:-${DATASET_ROOT}/metrics/sparse_action_vbench}"
NGPUS="${NGPUS:-8}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/score_sparse_action_dataset.py"

[[ "${NGPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "NGPUS must be positive." >&2; exit 2; }
[[ "${FORCE_RESCORE}" == "0" || "${FORCE_RESCORE}" == "1" ]] || {
  echo "FORCE_RESCORE must be 0 or 1." >&2
  exit 2
}
[[ -x "${VBENCH_PYTHON}" ]] || { echo "Python is not executable: ${VBENCH_PYTHON}" >&2; exit 1; }
[[ -f "${DRIVER}" ]] || { echo "Scoring driver not found: ${DRIVER}" >&2; exit 1; }
[[ -f "${DATASET_ROOT}/sparse_dataset_manifest.json" ]] || {
  echo "Finalized sparse dataset manifest not found: ${DATASET_ROOT}" >&2
  exit 1
}

args=(
  "${DRIVER}" "${MODE}"
  --dataset-root "${DATASET_ROOT}"
  --out-dir "${OUT_DIR}"
  --vbench-root "${VBENCH_ROOT}"
  --vbench-python "${VBENCH_PYTHON}"
  --ngpus "${NGPUS}"
)
[[ -n "${EXPECTED_VBENCH_COMMIT}" ]] && args+=(--expected-vbench-commit "${EXPECTED_VBENCH_COMMIT}")
[[ "${FORCE_RESCORE}" == "1" ]] && args+=(--force-rescore)

if [[ "${MODE}" == "score" || "${MODE}" == "all" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required for scoring." >&2; exit 1; }
  [[ -d "${VBENCH_ROOT}" ]] || { echo "VBench root not found: ${VBENCH_ROOT}" >&2; exit 1; }
fi

PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${VBENCH_PYTHON}" "${args[@]}"
