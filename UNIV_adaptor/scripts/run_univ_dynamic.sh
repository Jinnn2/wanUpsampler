#!/usr/bin/env bash
# M2 of REQUIREMENTS_V2_SPEC.md: dynamic-layer scoring with the frozen v2 checklist. No generation.
# Needs the fixed-layer root (plan.json + cache/) and the M1 environment (.venv_vlm3 with Qwen3-VL).
#   bash UNIV_adaptor/scripts/run_univ_dynamic.sh plan pilot   # 16 pilot prompts (or: plan all)
#   bash UNIV_adaptor/scripts/run_univ_dynamic.sh check        # one clip + one pair on one GPU, with timings
#   bash UNIV_adaptor/scripts/run_univ_dynamic.sh run          # vlm -> pair -> report on all GPUs
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
SCOPE="${2:-pilot}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
FIXED_ROOT="${FIXED_ROOT:-${REPO_ROOT}/outputs/fixed_metrics_v1}"
CHECKLIST="${CHECKLIST:-${REPO_ROOT}/prompts/univ_requirements_v2_frozen.jsonl}"
DYN_ROOT="${DYN_ROOT:-${REPO_ROOT}/outputs/dynamic_v1_${SCOPE}}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/score_dynamic.py"
PROMPTS=("${REPO_ROOT}/prompts/univ_controlled_factor_v1.jsonl" "${REPO_ROOT}/prompts/univ_controlled_factor_dev2.jsonl")
PILOT_IDS=(0 1 8 9 10 11 12 13 100 101 102 103 104 105 106 107)
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"

fan_out() {  # $1 = mode (vlm or pair); one worker per GPU, fails if any worker fails
  local mode="$1" pids=() i
  mkdir -p "${DYN_ROOT}/logs"
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${VLM_PYTHON}" "${DRIVER}" "${mode}" --out "${DYN_ROOT}" --model "${JUDGE_MODEL}" \
      --shard-index "${i}" --shard-count "${#GPUS[@]}" > "${DYN_ROOT}/logs/${mode}_${i}.log" 2>&1 &
    pids+=("$!")
  done
  trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
  local failed=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "worker ${i} failed: ${DYN_ROOT}/logs/${mode}_${i}.log" >&2; failed=1; }
  done
  return "${failed}"
}

case "${MODE}" in
  plan)
    ids=()
    [[ "${SCOPE}" == "pilot" ]] && ids=(--prompt-ids "${PILOT_IDS[@]}")
    "${BASE_PYTHON}" "${DRIVER}" plan --fixed-root "${FIXED_ROOT}" --checklist "${CHECKLIST}" \
      --prompts "${PROMPTS[@]}" --out "${DYN_ROOT}" "${ids[@]}"
    ;;
  check)
    # One clip and one pair in a scratch copy of the plan, so the timings say how long `run` will take.
    CHECK="${DYN_ROOT}/check"
    rm -rf "${CHECK}" && mkdir -p "${CHECK}" && cp "${DYN_ROOT}/plan.json" "${CHECK}/"
    "${VLM_PYTHON}" -c 'import transformers, torch; print("transformers", transformers.__version__, "torch", torch.__version__)'
    export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
    "${VLM_PYTHON}" - "${DRIVER}" "${CHECK}" "${JUDGE_MODEL}" <<'PY'
import importlib.util, json, sys, time
spec = importlib.util.spec_from_file_location("sd", sys.argv[1]); sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)
out, model = sys.argv[2], sys.argv[3]
value = sd.load_plan(out)
t = time.time(); judge = sd.Judge(model); print(f"model loaded in {time.time() - t:.0f}s")
fulls = sd.full_of(value)
accel = next(v for v in value["videos"] if v["case_id"] != sd.FULL)
full = fulls[(accel["source"], accel["group_id"])]
record = value["checklist_by_prompt"][accel["prompt_id"]]
qs = sd.vlm_questions(record)
print(f"prompt {accel['prompt_id']}: {len(qs)} VLM questions; accelerated case {accel['case_id']}")
frames = {}
for v in (full, accel):
    frames[v["case_id"]] = sd.decode(v["video_path"])
    t = time.time(); ans = sd.score_single(judge, *frames[v["case_id"]], qs)
    print(f"single {v['case_id']}: {time.time() - t:.1f}s")
    for q in qs:
        print(f"   {q['id']} {q['type']:<9} {ans[q['id']]['p_yes']:.3f}  {q['question']}")
t = time.time(); pr = sd.score_pair(judge, frames[accel["case_id"]], frames[full["case_id"]], qs)
print(f"pair (all {len(qs)} questions, both orders): {time.time() - t:.1f}s")
for q in qs:
    a = pr[q["id"]]; print(f"   {q['id']} full-first {a['p_full_first']:.3f}  accel-first {a['p_accel_first']:.3f}")
PY
    ;;
  vlm) fan_out vlm ;;
  pair) fan_out pair ;;
  report)
    "${BASE_PYTHON}" "${DRIVER}" report --out "${DYN_ROOT}" | tee "${DYN_ROOT}/report.txt"
    tar czf "${DYN_ROOT}_analysis.tgz" -C "$(dirname "${DYN_ROOT}")" "$(basename "${DYN_ROOT}")/plan.json" \
      "$(basename "${DYN_ROOT}")/vlm" "$(basename "${DYN_ROOT}")/pair" "$(basename "${DYN_ROOT}")/dynamic_by_question.csv" \
      "$(basename "${DYN_ROOT}")/dynamic_by_video.csv" "$(basename "${DYN_ROOT}")/report.txt"
    echo "Download: ${DYN_ROOT}_analysis.tgz"
    ;;
  run) bash "$0" vlm "${SCOPE}" && bash "$0" pair "${SCOPE}" && bash "$0" report "${SCOPE}" ;;
  status)
    echo "vlm: $(ls "${DYN_ROOT}/vlm" 2>/dev/null | wc -l) clips; pair: $(ls "${DYN_ROOT}/pair" 2>/dev/null | wc -l) pairs"
    ;;
  *)
    echo "Usage: bash $0 {plan|check|vlm|pair|report|run|status} [pilot|all]"
    echo "Output: ${DYN_ROOT}; judge: ${JUDGE_MODEL}"
    ;;
esac
