#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-draft-check}"
case "${MODE}" in
  draft-check|plan-probes|plan|check) ;;
  *) echo "Usage: $0 [draft-check|plan-probes|plan|check]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
PROTOCOL="${PROTOCOL:-${PROJECT_ROOT}/UNIV_adaptor/configs/univ_sparse_controller_pilot.json}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/prompts/univ_controller_pilot_500.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_sparse_controller_pilot}"
PLAN_PATH="${PLAN_PATH:-${OUT_ROOT}/collection_plan.json}"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/plan_univ_sparse_dataset.py"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${PROTOCOL}" ]] || { echo "Protocol not found: ${PROTOCOL}" >&2; exit 1; }
[[ -f "${DRIVER}" ]] || { echo "Planner not found: ${DRIVER}" >&2; exit 1; }

case "${MODE}" in
  draft-check)
    "${PYTHON_BIN}" "${DRIVER}" check-protocol \
      --protocol "${PROTOCOL}" \
      --allow-pending-probe
    ;;
  plan-probes)
    [[ -f "${PROMPTS_FILE}" ]] || { echo "Prompts not found: ${PROMPTS_FILE}" >&2; exit 1; }
    "${PYTHON_BIN}" "${DRIVER}" plan-probes \
      --protocol "${PROTOCOL}" \
      --prompts "${PROMPTS_FILE}" \
      --output "${PLAN_PATH}"
    ;;
  plan)
    [[ -f "${PROMPTS_FILE}" ]] || { echo "Prompts not found: ${PROMPTS_FILE}" >&2; exit 1; }
    "${PYTHON_BIN}" "${DRIVER}" plan \
      --protocol "${PROTOCOL}" \
      --prompts "${PROMPTS_FILE}" \
      --output "${PLAN_PATH}"
    ;;
  check)
    [[ -f "${PLAN_PATH}" ]] || { echo "Plan not found: ${PLAN_PATH}" >&2; exit 1; }
    "${PYTHON_BIN}" "${DRIVER}" check --plan "${PLAN_PATH}"
    ;;
esac
