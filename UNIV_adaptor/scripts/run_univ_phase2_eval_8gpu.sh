#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-all}"
case "$MODE" in locate|check|score|report|all) ;; *) echo "Usage: $0 [locate|check|score|report|all]" >&2; exit 2;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-}"
if [[ -z "$WAN_PYTHON" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then WAN_PYTHON=/opt/conda/bin/python
  else WAN_PYTHON="$(command -v python || command -v python3)"; fi
fi
DRIVER="$PROJECT_ROOT/UNIV_adaptor/scripts/data/score_phase2_dataset.py"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "$MODE" == locate ]]; then
  LOCATE_ARGS=(locate)
  [[ -z "${OUT_ROOT:-}" ]] || LOCATE_ARGS+=(--dataset-root "$OUT_ROOT")
  if [[ -n "${SEARCH_ROOT:-}" ]]; then LOCATE_ARGS+=(--search-root "$SEARCH_ROOT")
  elif [[ -z "${OUT_ROOT:-}" ]]; then LOCATE_ARGS+=(--search-root "$PROJECT_ROOT/outputs"); fi
  exec "$WAN_PYTHON" "$DRIVER" "${LOCATE_ARGS[@]}"
fi
# Several phase2 roots may coexist; never silently select a different run.
: "${OUT_ROOT:?Set OUT_ROOT to the Phase2 directory whose finalize completed; run this script with locate to inspect candidates}"
[[ -f "$OUT_ROOT/generation_manifest.json" && -f "$OUT_ROOT/collection_plan.json" ]] || {
  echo "Missing Phase2 manifest/plan under OUT_ROOT=$OUT_ROOT. Run: bash $0 locate" >&2; exit 1;
}
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-}"
EVAL_OUT="${EVAL_OUT:-$OUT_ROOT/metrics/phase2_quality}"
ARGS=("$MODE" --dataset-root "$OUT_ROOT" --out-dir "$EVAL_OUT" --tie-epsilon "${TIE_EPSILON:-0.001}")
if [[ -n "${BUDGETS_SECONDS:-}" ]]; then
  read -r -a BUDGET_ARRAY <<< "$BUDGETS_SECONDS"
  ARGS+=(--budgets-seconds "${BUDGET_ARRAY[@]}")
fi
if [[ "$MODE" == check || "$MODE" == report ]]; then
  exec "$WAN_PYTHON" "$DRIVER" "${ARGS[@]}"
fi
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ ${#GPUS[@]} -eq 8 ]] || { echo "Expected eight distinct numeric GPU_IDS" >&2; exit 2; }
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z "${SEEN[$gpu]:-}" ]] || { echo "Invalid/duplicate GPU id: $gpu" >&2; exit 2; }
  SEEN[$gpu]=1
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
[[ -f "$VBENCH_ROOT/evaluate.py" ]] || { echo "Missing VBench checkout: $VBENCH_ROOT" >&2; exit 1; }
probe_python() {
  (cd "$VBENCH_ROOT"; "$1" -c 'import torch, vbench; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 8')
}
if [[ -n "$VBENCH_PYTHON" ]]; then
  probe_python "$VBENCH_PYTHON" || { echo "Explicit VBENCH_PYTHON failed: $VBENCH_PYTHON" >&2; exit 1; }
else
  for candidate in "$WAN_PYTHON" /opt/conda/envs/vbench/bin/python /opt/conda/bin/python "$(command -v python 2>/dev/null || true)"; do
    [[ -n "$candidate" ]] || continue
    if probe_python "$candidate" >/dev/null 2>&1; then VBENCH_PYTHON="$candidate"; break; fi
  done
  [[ -n "$VBENCH_PYTHON" ]] || {
    echo "No Python can import torch/vbench and see eight CUDA GPUs. Set VBENCH_PYTHON to your working evaluation environment." >&2; exit 1;
  }
fi
[[ "${FORCE_RESCORE:-0}" != 1 ]] || ARGS+=(--force-rescore)
mkdir -p "$EVAL_OUT"
echo "[Phase2 evaluation] root=$OUT_ROOT python=$VBENCH_PYTHON GPUs=$GPU_IDS"
"$VBENCH_PYTHON" "$DRIVER" "${ARGS[@]}" --vbench-root "$VBENCH_ROOT" --vbench-python "$VBENCH_PYTHON" --ngpus 8 2>&1 | tee -a "$EVAL_OUT/evaluation.log"
