#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export MODEL_TYPE=all
export OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_b4_hybrid_selection}"

exec bash "${SCRIPT_DIR}/run_multiseed_variable_lambda_selection.sh"
