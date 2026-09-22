#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
CONTROLLED_FACTOR_ROOT="${CONTROLLED_FACTOR_ROOT:-${PROJECT_ROOT}/outputs/univ_controlled_factor_v1}"
CONTROLLED_FACTOR_SCORED_DIR="${CONTROLLED_FACTOR_SCORED_DIR:-${CONTROLLED_FACTOR_ROOT}/metrics/controlled_factor_vbench}"
CONTROLLED_FACTOR_AUDIT_OUT="${CONTROLLED_FACTOR_AUDIT_OUT:-${CONTROLLED_FACTOR_SCORED_DIR}/seed_value_audit_v2}"
"${WAN_PYTHON}" "${SCRIPT_DIR}/data/audit_controlled_factor_seed_value.py" \
  --scored-dir "${CONTROLLED_FACTOR_SCORED_DIR}" \
  --out-dir "${CONTROLLED_FACTOR_AUDIT_OUT}" "$@"
archive_files=(
  audit.json audit_request.json oracle_summary.csv decisions_by_seed.csv
  pair_variance.csv margin_summary.csv policy_by_held_seed.csv report.md
)
[[ -f "${CONTROLLED_FACTOR_AUDIT_OUT}/learned_summary.csv" ]] && archive_files+=(learned_summary.csv)
tar -czf "${CONTROLLED_FACTOR_AUDIT_OUT}/seed_value_audit.tgz" \
  -C "${CONTROLLED_FACTOR_AUDIT_OUT}" "${archive_files[@]}"
echo "Send back: ${CONTROLLED_FACTOR_AUDIT_OUT}/seed_value_audit.tgz"
