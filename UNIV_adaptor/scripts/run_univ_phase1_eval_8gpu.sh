#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-all}"
case "$MODE" in check|score|report|all) ;; *) echo "Usage: $0 [check|score|report|all]" >&2; exit 2;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_prompt_budget_phase1_pilot_20260917}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/envs/vbench/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ ${#GPUS[@]} -eq 8 ]] || { echo "Expected eight GPU_IDS" >&2; exit 2; }
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/score_phase1_dataset.py"
"$WAN_PYTHON" "${PROJECT_ROOT}/UNIV_adaptor/scripts/data/analyze_phase1_runtime.py" --dataset-root "$OUT_ROOT" --out-dir "$OUT_ROOT/metrics/runtime"
if [[ "$MODE" == check || "$MODE" == report ]]; then
  "$WAN_PYTHON" "$DRIVER" "$MODE" --dataset-root "$OUT_ROOT"
  exit
fi
(cd "$VBENCH_ROOT"; "$VBENCH_PYTHON" -c 'import torch, vbench; assert torch.cuda.is_available()')
mkdir -p "$OUT_ROOT/metrics/phase1_quality"
LOCK="$OUT_ROOT/metrics/phase1_quality/.scoring.lock"
mkdir "$LOCK" || { echo "Scoring already running or stale lock: $LOCK" >&2; exit 1; }
trap 'rmdir "$LOCK"' EXIT
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$VBENCH_PYTHON" "$DRIVER" "$MODE" --dataset-root "$OUT_ROOT" --vbench-root "$VBENCH_ROOT" --vbench-python "$VBENCH_PYTHON" --ngpus 8 2>&1 | tee -a "$OUT_ROOT/metrics/phase1_quality/evaluation.log"
