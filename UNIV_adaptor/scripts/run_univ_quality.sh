#!/usr/bin/env bash
# Visual-quality layer (score_quality.py): prompt-independent sharpness / artifact / aesthetic / temporal /
# acceptable judgments with the dynamic layer's Qwen3-VL judge, plus optional DOVER. No generation.
# Needs the fixed-layer root (plan.json + cache/) and the M1 environment (.venv_vlm3).
#   bash UNIV_adaptor/scripts/run_univ_quality.sh plan
#   bash UNIV_adaptor/scripts/run_univ_quality.sh check     # one clip + one pair on one GPU, with timings
#   bash UNIV_adaptor/scripts/run_univ_quality.sh run       # vlm -> pair -> report on all GPUs
#   DOVER_ROOT=/path/to/DOVER bash UNIV_adaptor/scripts/run_univ_quality.sh dover   # optional, then report again
#   bash UNIV_adaptor/scripts/run_univ_quality.sh v2        # + fixed-layer judder/flicker -> ${Q_ROOT}_v2/
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
DOVER_PYTHON="${DOVER_PYTHON:-${VLM_PYTHON}}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
FIXED_ROOT="${FIXED_ROOT:-${REPO_ROOT}/outputs/fixed_metrics_v1}"
Q_ROOT="${Q_ROOT:-${REPO_ROOT}/outputs/quality_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/score_quality.py"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"

fan_out() {  # $1 = mode (vlm, pair or dover); one worker per GPU, fails if any worker fails
  local mode="$1" pids=() i extra
  mkdir -p "${Q_ROOT}/logs"
  for i in "${!GPUS[@]}"; do
    if [[ "${mode}" == "dover" ]]; then
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${DOVER_PYTHON}" "${DRIVER}" dover --out "${Q_ROOT}" --dover-root "${DOVER_ROOT:?set DOVER_ROOT}" \
        --shard-index "${i}" --shard-count "${#GPUS[@]}" > "${Q_ROOT}/logs/dover_${i}.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${VLM_PYTHON}" "${DRIVER}" "${mode}" --out "${Q_ROOT}" --model "${JUDGE_MODEL}" \
        --shard-index "${i}" --shard-count "${#GPUS[@]}" > "${Q_ROOT}/logs/${mode}_${i}.log" 2>&1 &
    fi
    pids+=("$!")
  done
  trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
  local failed=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "worker ${i} failed: ${Q_ROOT}/logs/${mode}_${i}.log" >&2; failed=1; }
  done
  return "${failed}"
}

case "${MODE}" in
  plan)
    "${BASE_PYTHON}" "${DRIVER}" plan --fixed-root "${FIXED_ROOT}" --out "${Q_ROOT}"
    ;;
  check)
    export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
    "${VLM_PYTHON}" - "${DRIVER}" "${Q_ROOT}" "${JUDGE_MODEL}" <<'PY'
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("sq", sys.argv[1]); sq = importlib.util.module_from_spec(spec); spec.loader.exec_module(sq)
value = sq.load_plan(sys.argv[2])
t = time.time(); judge = sq.Judge(sys.argv[3]); print(f"model loaded in {time.time() - t:.0f}s")
# The worst-looking arm in the pilot (LR_BICUBIC) against its FULL clip: the scores should differ visibly.
accel = next(v for v in value["videos"] if v["case_id"] == "LR_BICUBIC")
full = sq.full_of(value)[(accel["source"], accel["group_id"])]
frames = {v["case_id"]: sq.verified(v) for v in (full, accel)}
for case, fr in frames.items():
    t = time.time(); ans = sq.score_single(judge, *fr)
    print(f"single {case}: {time.time() - t:.1f}s  " + "  ".join(f"{a} {ans[a]['p_yes']:.3f}" for a in sq.ASPECTS))
t = time.time(); pr = sq.score_pair(judge, frames[accel["case_id"]], frames[full["case_id"]])
print(f"pair: {time.time() - t:.1f}s")
for a in sq.ASPECTS:
    print(f"   {a:<10} full-first {pr[a]['p_full_first']:.3f}  accel-first {pr[a]['p_accel_first']:.3f}")
PY
    ;;
  check-dover)
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${DOVER_PYTHON}" - "${DRIVER}" "${Q_ROOT}" "${DOVER_ROOT:?set DOVER_ROOT}" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("sq", sys.argv[1]); sq = importlib.util.module_from_spec(spec); spec.loader.exec_module(sq)
value = sq.load_plan(sys.argv[2]); model = sq.Dover(sys.argv[3])
accel = next(v for v in value["videos"] if v["case_id"] == "LR_BICUBIC")
full = sq.full_of(value)[(accel["source"], accel["group_id"])]
for v in (full, accel):
    print(v["case_id"], {k: round(x, 4) for k, x in model(v["video_path"]).items()})
PY
    ;;
  vlm) fan_out vlm ;;
  pair) fan_out pair ;;
  dover) fan_out dover ;;
  report)
    "${BASE_PYTHON}" "${DRIVER}" report --out "${Q_ROOT}" | tee "${Q_ROOT}/report.txt"
    parts=(plan.json vlm pair quality_by_video.csv report.txt)
    [[ -d "${Q_ROOT}/dover" ]] && parts+=(dover)
    tar czf "${Q_ROOT}_analysis.tgz" -C "$(dirname "${Q_ROOT}")" "${parts[@]/#/$(basename "${Q_ROOT}")/}"
    echo "Download: ${Q_ROOT}_analysis.tgz"
    ;;
  run) bash "$0" vlm && bash "$0" pair && bash "$0" report ;;
  v2)
    # Adds the fixed-layer judder/flicker checks (quality_v2.py); thresholds from the FULL clips of CALIB_FIXED.
    mkdir -p "${Q_ROOT}_v2"
    "${BASE_PYTHON}" "${REPO_ROOT}/UNIV_adaptor/scripts/data/quality_v2.py" --quality "${Q_ROOT}/quality_by_video.csv" \
      --fixed "${FIXED_ROOT}/fixed_by_video.csv" --calibration-fixed "${CALIB_FIXED:-${FIXED_ROOT}/fixed_by_video.csv}" \
      --out "${Q_ROOT}_v2/quality_by_video.csv" | tee "${Q_ROOT}_v2/report.txt"
    ;;
  status)
    echo "vlm: $(ls "${Q_ROOT}/vlm" 2>/dev/null | wc -l) clips; pair: $(ls "${Q_ROOT}/pair" 2>/dev/null | wc -l) pairs;" \
      "dover: $(ls "${Q_ROOT}/dover" 2>/dev/null | wc -l) clips"
    ;;
  *)
    echo "Usage: bash $0 {plan|check|check-dover|vlm|pair|dover|report|run|status}"
    echo "Output: ${Q_ROOT}; judge: ${JUDGE_MODEL}"
    ;;
esac
