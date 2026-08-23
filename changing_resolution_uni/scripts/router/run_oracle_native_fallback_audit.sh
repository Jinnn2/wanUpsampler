#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/oracle_native_fallback_audit_${RUN_ID}}"
SAMPLE_COUNT="${SAMPLE_COUNT:-12}"
LAMBDA_VALUE="${LAMBDA_VALUE:-0.01}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/changing_resolution_uni/scripts/data/audit_oracle_native_fallback.py" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --sample-count "${SAMPLE_COUNT}" \
  --lambda-value "${LAMBDA_VALUE}"

archive_path="${OUT_DIR}.tar.gz"
tar -C "$(dirname "${OUT_DIR}")" -czf "${archive_path}" "$(basename "${OUT_DIR}")"

echo "Audit directory: ${OUT_DIR}"
echo "Upload this archive for analysis: ${archive_path}"
