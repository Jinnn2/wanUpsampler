#!/usr/bin/env bash
set -euo pipefail

# Sweep intermediate last-step-skip LoRA checkpoints on the 480p Phase-1
# objective. For each checkpoint, this runs:
#   original3 clean-pred | lora3 clean-pred | teacher4
# then computes closeness metrics against teacher4 and writes an aggregate CSV.
#
# Useful overrides:
#   CR_DISTILL_LORA_OUT_DIR=outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3
#   CHECKPOINTS="step_0004000.safetensors step_0005000.safetensors latest.safetensors"
#   CANDIDATE_STEPS="3000 4000 5000 6000 8000 10000"
#   SWEEP_OUT_ROOT=outputs/eval_lora_ckpt_sweep_480p
#   LIMIT=10 SEED=42 CUDA_VISIBLE_DEVICES=0 SKIP_COMPLETED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

CR_DISTILL_LORA_OUT_DIR="${CR_DISTILL_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3}"
SWEEP_OUT_ROOT="${SWEEP_OUT_ROOT:-${PROJECT_ROOT}/outputs/eval_lora_ckpt_sweep_480p}"
CANDIDATE_STEPS="${CANDIDATE_STEPS:-3000 4000 5000 6000 8000 10000}"
CHECKPOINTS="${CHECKPOINTS:-}"
METRICS="${METRICS:-l1 mse psnr ssim temporal_l1}"
PRIMARY_METRIC="${PRIMARY_METRIC:-l1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "${SWEEP_OUT_ROOT}"

resolve_checkpoint() {
  local item="$1"
  if [[ "${item}" == /* ]]; then
    echo "${item}"
    return
  fi
  if [[ "${item}" == *.safetensors ]]; then
    if [[ -f "${item}" ]]; then
      echo "${item}"
    else
      echo "${CR_DISTILL_LORA_OUT_DIR}/${item}"
    fi
    return
  fi
  if [[ "${item}" == "latest" ]]; then
    echo "${CR_DISTILL_LORA_OUT_DIR}/latest.safetensors"
    return
  fi
  printf "%s/step_%07d.safetensors\n" "${CR_DISTILL_LORA_OUT_DIR}" "${item}"
}

checkpoint_tag() {
  local path="$1"
  local name
  name="$(basename "${path}")"
  name="${name%.safetensors}"
  echo "${name//[^A-Za-z0-9_.-]/_}"
}

build_checkpoint_list() {
  if [[ -n "${CHECKPOINTS}" ]]; then
    printf "%s\n" ${CHECKPOINTS}
    return
  fi
  printf "%s\n" ${CANDIDATE_STEPS}
  printf "%s\n" latest
}

run_one_checkpoint() {
  local item="$1"
  local ckpt
  ckpt="$(resolve_checkpoint "${item}")"
  local tag
  tag="$(checkpoint_tag "${ckpt}")"
  local out_root="${SWEEP_OUT_ROOT}/${tag}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[sweep] skip missing checkpoint: ${ckpt}" >&2
    return
  fi

  echo ""
  echo "[sweep] checkpoint=${tag}"
  echo "  ckpt    : ${ckpt}"
  echo "  out_root: ${out_root}"

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${out_root}/metrics/original_lora_teacher_summary.csv" ]]; then
    echo "  skip    : summary already exists"
    return
  fi

  LORA_CKPT="${ckpt}" \
  OUT_ROOT="${out_root}" \
  bash "${SCRIPT_DIR}/run_last_step_skip_lora_clean_pred_compare.sh"

  OUT_ROOT="${out_root}" \
  bash "${SCRIPT_DIR}/run_last_step_skip_lora_video_closeness_eval.sh" \
    --metrics ${METRICS} \
    --primary_metric "${PRIMARY_METRIC}"
}

aggregate_summaries() {
  local aggregate_csv="${SWEEP_OUT_ROOT}/checkpoint_metric_summary.csv"
  local rank_csv="${SWEEP_OUT_ROOT}/checkpoint_rank_by_${PRIMARY_METRIC}.csv"

  python - "${SWEEP_OUT_ROOT}" "${PRIMARY_METRIC}" "${aggregate_csv}" "${rank_csv}" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
primary_metric = sys.argv[2]
aggregate_csv = Path(sys.argv[3])
rank_csv = Path(sys.argv[4])

rows = []
rank_rows = []
for summary_path in sorted(root.glob("*/metrics/original_lora_teacher_summary.csv")):
    checkpoint = summary_path.parent.parent.name
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["checkpoint"] = checkpoint
            row["summary_csv"] = str(summary_path)
            rows.append(row)
            if row.get("metric") == primary_metric:
                try:
                    delta = float(row["delta_lora_minus_original_mean"])
                    win_rate = float(row["lora_win_rate"])
                    lora_mean = float(row["lora_mean"])
                except (KeyError, TypeError, ValueError):
                    delta = math.nan
                    win_rate = math.nan
                    lora_mean = math.nan
                rank_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "metric": primary_metric,
                        "better": row.get("better", ""),
                        "lora_mean": row.get("lora_mean", ""),
                        "original_mean": row.get("original_mean", ""),
                        "delta_lora_minus_original_mean": row.get("delta_lora_minus_original_mean", ""),
                        "lora_win_rate": row.get("lora_win_rate", ""),
                        "_delta": delta,
                        "_win_rate": win_rate,
                        "_lora_mean": lora_mean,
                        "summary_csv": str(summary_path),
                    }
                )

if rows:
    columns = ["checkpoint"] + [key for key in rows[0].keys() if key != "checkpoint"]
    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def sort_key(row):
    # For the delta field, lower is better metrics improve when delta < 0,
    # higher is better metrics improve when delta > 0. The existing summary
    # stores the direction in the "better" column.
    better = row.get("better", "")
    delta = row.get("_delta", math.nan)
    win_rate = row.get("_win_rate", math.nan)
    lora_mean = row.get("_lora_mean", math.nan)
    if not math.isfinite(delta):
        delta_score = math.inf
    elif better == "lower":
        delta_score = delta
    else:
        delta_score = -delta
    if not math.isfinite(lora_mean):
        lora_score = math.inf
    elif better == "lower":
        lora_score = lora_mean
    else:
        lora_score = -lora_mean
    return (delta_score, -win_rate if math.isfinite(win_rate) else math.inf, lora_score)

rank_rows.sort(key=sort_key)
for row in rank_rows:
    row.pop("_delta", None)
    row.pop("_win_rate", None)
    row.pop("_lora_mean", None)

if rank_rows:
    columns = list(rank_rows[0].keys())
    with rank_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rank_rows)

print(f"aggregate_csv={aggregate_csv}")
print(f"rank_csv={rank_csv}")
print(f"checkpoints={len(rank_rows)}")
PY
}

echo "[sweep] checkpoint_dir=${CR_DISTILL_LORA_OUT_DIR}"
echo "[sweep] out_root=${SWEEP_OUT_ROOT}"
echo "[sweep] metrics=${METRICS}"
echo "[sweep] primary_metric=${PRIMARY_METRIC}"
echo "[sweep] skip_completed=${SKIP_COMPLETED}"

while IFS= read -r item; do
  [[ -z "${item}" ]] && continue
  run_one_checkpoint "${item}"
done < <(build_checkpoint_list)

aggregate_summaries

echo ""
echo "[sweep] done"
echo "  aggregate: ${SWEEP_OUT_ROOT}/checkpoint_metric_summary.csv"
echo "  ranking  : ${SWEEP_OUT_ROOT}/checkpoint_rank_by_${PRIMARY_METRIC}.csv"
