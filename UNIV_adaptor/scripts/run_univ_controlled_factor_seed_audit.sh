#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/outputs/univ_controlled_factor_v1}"
SCORED_DIR="${SCORED_DIR:-${DATASET_ROOT}/metrics/controlled_factor_vbench}"
OUT_DIR="${OUT_DIR:-${SCORED_DIR}/seed_value_audit}"
"${WAN_PYTHON}" "${SCRIPT_DIR}/data/audit_controlled_factor_seed_value.py" \
  --scored-dir "${SCORED_DIR}" --out-dir "${OUT_DIR}" "$@"
tar -czf "${OUT_DIR}/seed_value_audit.tgz" -C "${OUT_DIR}" \
  audit.json audit_request.json oracle_summary.csv decisions_by_seed.csv \
  pair_variance.csv margin_summary.csv policy_by_held_seed.csv report.md
echo "Send back: ${OUT_DIR}/seed_value_audit.tgz"
