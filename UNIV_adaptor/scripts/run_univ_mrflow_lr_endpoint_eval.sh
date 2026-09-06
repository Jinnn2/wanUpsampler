#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/univ_mrflow_lr_endpoint_v1}"
exec bash "${PROJECT_ROOT}/UNIV_adaptor/scripts/run_univ_mrflow_refinement_eval.sh" "${1:-run}"
