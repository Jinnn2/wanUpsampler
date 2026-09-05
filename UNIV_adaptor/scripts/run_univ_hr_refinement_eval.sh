#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
case "${MODE}" in check|run) ;; *) echo "Usage: $0 [check|run]" >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/univ_hr_refinement_ablation_v1}"
[[ -f "${OUT_DIR}/comparison_summary.json" ]] || { echo "Missing comparison_summary.json under ${OUT_DIR}" >&2; exit 1; }
[[ -f "${VBENCH_ROOT}/evaluate.py" ]] || { echo "Missing VBench evaluate.py under ${VBENCH_ROOT}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONPATH="${VBENCH_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${VBENCH_PYTHON}" ]]; then
  for candidate in /opt/conda/envs/vbench/bin/python /opt/conda/bin/python "$(command -v python 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if (cd "${VBENCH_ROOT}"; "${candidate}" -c 'import torch, vbench' >/dev/null 2>&1); then
      VBENCH_PYTHON="${candidate}"
      break
    fi
  done
fi
[[ -n "${VBENCH_PYTHON}" && -x "${VBENCH_PYTHON}" ]] || {
  echo "Set VBENCH_PYTHON to an environment with torch and VBench installed." >&2; exit 1;
}
(cd "${VBENCH_ROOT}"; "${VBENCH_PYTHON}" -c 'import torch, vbench; assert torch.cuda.is_available(), "VBench CUDA unavailable"')
echo "VBench Python: ${VBENCH_PYTHON}"
args=(
  "${PROJECT_ROOT}/UNIV_adaptor/scripts/validation/score_hr_refinement_ablation.py" "${MODE}"
  --out-dir "${OUT_DIR}" --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}"
)
[[ -n "${VBENCH_COMMIT:-}" ]] && args+=(--vbench-commit "${VBENCH_COMMIT}")
[[ "${FORCE_VBENCH:-0}" == "1" ]] && args+=(--force)
exec "${VBENCH_PYTHON}" "${args[@]}"
