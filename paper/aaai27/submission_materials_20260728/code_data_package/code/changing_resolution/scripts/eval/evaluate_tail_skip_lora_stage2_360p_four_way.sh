#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_stage2_four_way_360p}"
RESULT_DIR="${RESULT_DIR:-${OUT_ROOT}/evaluation}"
METRIC_DEVICE_ARGS=()
if [[ "${CPU:-0}" == "1" ]]; then
  METRIC_DEVICE_ARGS=(--cpu)
fi
METRICS=(l1 mse psnr ssim temporal_l1)
if [[ "${ENABLE_LPIPS:-1}" == "1" ]]; then
  METRICS+=(lpips)
fi

for case_name in ori45_stage2 lora45_stage2 teacher50_interp teacher50_stage2; do
  case_dir="${OUT_ROOT}/videos/${case_name}"
  if [[ ! -d "${case_dir}" ]]; then
    echo "Case directory not found: ${case_dir}" >&2
    exit 1
  fi
done

mkdir -p "${RESULT_DIR}"

run_lora_metrics() {
  local teacher_case="$1"
  local name_prefix="$2"
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/eval_last_step_skip_lora_video_closeness.py" \
  --out_root "${OUT_ROOT}" \
  --result_dir "${RESULT_DIR}" \
  --original_case ori45_stage2 \
  --lora_case lora45_stage2 \
  --teacher_case "${teacher_case}" \
  --metrics "${METRICS[@]}" \
  --primary_metric l1 \
  --metric_batch_size "${METRIC_BATCH_SIZE:-2}" \
  --frame_stride "${FRAME_STRIDE:-1}" \
  --limit "${LIMIT:-10}" \
  --jsonl_name "${name_prefix}_metrics.jsonl" \
  --csv_name "${name_prefix}_metrics.csv" \
  --summary_json_name "${name_prefix}_summary.json" \
  --summary_csv_name "${name_prefix}_summary.csv" \
  "${METRIC_DEVICE_ARGS[@]}"
}

# Primary comparison: the reference uses the same new Stage2 operator as both
# step-45 candidates, so the measured difference isolates the LoRA handoff.
run_lora_metrics teacher50_stage2 lora_vs_teacher_stage2

# Secondary robustness comparison: use interpolation-based teacher50 as a
# second anchor. It is not the primary score, but guards against conclusions
# that only hold when the reference shares the learned Stage2 operator.
run_lora_metrics teacher50_interp lora_vs_teacher_interp

export OUT_ROOT RESULT_DIR
python - <<'PY'
import csv
import os
import re
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
result_dir = Path(os.environ["RESULT_DIR"])
output = result_dir / "human_pairwise_review.csv"

pattern = re.compile(r"^(?P<case>.+)_(?P<index>\d+)_seed(?P<seed>-?\d+)\.mp4$")

def index_case(case_name):
    result = {}
    for path in sorted((out_root / "videos" / case_name).glob("*.mp4")):
        match = pattern.match(path.name)
        if match and match.group("case") == case_name:
            result[(int(match.group("index")), int(match.group("seed")))] = path
    return result

comparisons = [
    ("lora_vs_ori_fullchain", "ori45_stage2", "lora45_stage2"),
    ("stage2_vs_interp_teacher50", "teacher50_interp", "teacher50_stage2"),
]
rows = []
for comparison, left_case, right_case in comparisons:
    left = index_case(left_case)
    right = index_case(right_case)
    for sample_index, seed in sorted(set(left) & set(right)):
        rows.append(
            {
                "comparison": comparison,
                "sample_index": sample_index,
                "seed": seed,
                "left_case": left_case,
                "right_case": right_case,
                "detail_winner": "",
                "artifact_cleanliness_winner": "",
                "temporal_stability_winner": "",
                "structure_identity_winner": "",
                "overall_winner": "",
                "confidence_1_to_5": "",
                "severe_failure_case": "",
                "notes": "",
            }
        )

columns = list(rows[0]) if rows else []
if output.exists():
    print(f"Human review template already exists; preserved: {output}")
elif not rows:
    raise SystemExit("No matched videos found for human-review template")
else:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Human review template: {output}")
PY

echo "Evaluation protocol: ${PROJECT_ROOT}/doc/360P_FOUR_WAY_EVAL_PROTOCOL.md"
echo "Primary summary   : ${RESULT_DIR}/lora_vs_teacher_stage2_summary.csv"
echo "Second anchor     : ${RESULT_DIR}/lora_vs_teacher_interp_summary.csv"
echo "Human review CSV  : ${RESULT_DIR}/human_pairwise_review.csv"
