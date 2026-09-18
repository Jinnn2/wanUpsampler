#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-all}"
case "$MODE" in check|embed|train|all) ;; *) echo "Usage: $0 [check|embed|train|all]" >&2; exit 2;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -z "${QUALITY_DIR:-}" ]]; then
  : "${OUT_ROOT:?Set QUALITY_DIR to phase2_quality or OUT_ROOT to the completed generation root}"
  QUALITY_DIR="$OUT_ROOT/metrics/phase2_quality"
fi
WAN_PYTHON="${WAN_PYTHON:-}"
if [[ -z "$WAN_PYTHON" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then WAN_PYTHON=/opt/conda/bin/python
  else WAN_PYTHON="$(command -v python || command -v python3)"; fi
fi
FEATURES="${FEATURES:-tfidf}"
ACTION_SET="${ACTION_SET:-three}"
PRIOR_OUT="${PRIOR_OUT:-$QUALITY_DIR/gain_prior_${FEATURES}_${ACTION_SET}}"
T5_DIR="${T5_DIR:-$QUALITY_DIR/t5_phase2}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
export LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
export PYTHONPATH="$PROJECT_ROOT:$LIGHTX2V_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
DRIVER="$PROJECT_ROOT/UNIV_adaptor/scripts/router/train_phase2_gain_prior.py"
ARGS=(--quality-dir "$QUALITY_DIR" --out-dir "$PRIOR_OUT" --features "$FEATURES" --action-set "$ACTION_SET"
      --t5-dir "$T5_DIR" --model-root "$MODEL_ROOT")
if [[ -n "${BUDGETS_SECONDS:-}" ]]; then
  read -r -a BUDGET_ARRAY <<< "$BUDGETS_SECONDS"
  ARGS+=(--budgets-seconds "${BUDGET_ARRAY[@]}")
fi
if [[ "$MODE" == check ]]; then exec "$WAN_PYTHON" "$DRIVER" check "${ARGS[@]}"; fi
"$WAN_PYTHON" "$DRIVER" check "${ARGS[@]}"
if [[ "$MODE" == embed || ( "$MODE" == all && "$FEATURES" == t5 ) ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "$WAN_PYTHON" "$DRIVER" embed "${ARGS[@]}"
fi
if [[ "$MODE" == train || "$MODE" == all ]]; then
  "$WAN_PYTHON" "$DRIVER" train "${ARGS[@]}"
fi
