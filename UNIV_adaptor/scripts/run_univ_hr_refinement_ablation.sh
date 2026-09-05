#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"
case "${MODE}" in plan|check|run) ;; *) echo "Usage: $0 [plan|check|run]" >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
export LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
export DTYPE="${DTYPE:-BF16}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
args=(
  "${PROJECT_ROOT}/UNIV_adaptor/scripts/validation/run_hr_refinement_ablation.py" "${MODE}"
  --out-dir "${OUT_DIR:-${PROJECT_ROOT}/outputs/univ_hr_refinement_ablation_v1}"
  --seed "${SEED:-42}"
)
[[ -n "${PROMPT:-}" ]] && args+=(--prompt "${PROMPT}")
[[ "${NEGATIVE_PROMPT+x}" == "x" ]] && args+=(--negative-prompt "${NEGATIVE_PROMPT}")
[[ -n "${TEMPLATE_CONFIG:-}" ]] && args+=(--template-config "${TEMPLATE_CONFIG}")
exec "${WAN_PYTHON}" "${args[@]}"
