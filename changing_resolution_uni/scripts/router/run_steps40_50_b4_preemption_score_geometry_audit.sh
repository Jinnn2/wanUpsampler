#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENERATION_ROOT="${GENERATION_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1500_8gpu}"
DATASET_DIR="${DATASET_DIR:-${GENERATION_ROOT}/router_variable_lambda_states_selection_20260829_h100_profile_v1}"
B4_RUNS_ROOT="${B4_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_variable_lambda_1500_steps40_50_overall_v1/b4_residual}"
VERIFIER_RUNS_ROOT="${VERIFIER_RUNS_ROOT:-${PROJECT_ROOT}/outputs/router_steps40_50_b4_preemption_verifier_v2}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/router_steps40_50_b4_preemption_score_geometry_audit_v1}"
DEVICE="${DEVICE:-cuda}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4096}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2041}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "Missing prepared state dataset: ${DATASET_DIR}" >&2
  echo "Prepared datasets under ${GENERATION_ROOT}:" >&2
  find "${GENERATION_ROOT}" -mindepth 2 -maxdepth 2 -name dataset_manifest.json -printf '  %h\n' >&2
  exit 2
fi
if [[ ! -d "${B4_RUNS_ROOT}" ]]; then
  echo "Missing frozen steps40-50 B4 suite: ${B4_RUNS_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${VERIFIER_RUNS_ROOT}" ]]; then
  echo "Missing V2 preemption verifier suite: ${VERIFIER_RUNS_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUT_DIR}" >&2
  exit 2
fi

LATENCY_PROFILE_SHA256="${EXPECTED_LATENCY_PROFILE_SHA256:-$(python - "${DATASET_DIR}/dataset_manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = str(manifest.get("latency_profile", {}).get("sha256", ""))
if len(value) != 64:
    raise SystemExit("Dataset manifest has no locked latency profile SHA256")
print(value)
PY
)}"

python -m unittest \
  changing_resolution_uni.scripts.router.tests.test_b4_preemption_score_geometry

python "${SCRIPT_DIR}/audit_b4_preemption_score_geometry.py" \
  --dataset-dir "${DATASET_DIR}" \
  --b4-runs-root "${B4_RUNS_ROOT}" \
  --verifier-runs-root "${VERIFIER_RUNS_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --expected-latency-profile-sha256 "${LATENCY_PROFILE_SHA256}" \
  --device "${DEVICE}"

echo "Frozen preemption score geometry audit: ${OUT_DIR}/score_geometry_report.json"
echo "No weights, videos, latent archives, or test records were modified or accessed."
