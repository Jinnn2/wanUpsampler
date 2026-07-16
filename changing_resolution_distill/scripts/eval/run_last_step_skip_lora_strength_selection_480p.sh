#!/usr/bin/env bash
set -euo pipefail

# Coarse validation sweep for one fixed Distill 480p LoRA checkpoint.
# original3 and teacher4 are generated once, then hard-linked into each
# strength run. Only the LoRA branch is regenerated for every strength.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LORA_CKPT="${LORA_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/step_0010000.safetensors}"
CHECKPOINT_TAG="$(basename "${LORA_CKPT}" .safetensors)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/aaai27_experiments/distill_lora_strength_selection_480p_${CHECKPOINT_TAG}}"
STRENGTHS="${STRENGTHS:-0.0 0.25 0.5 0.75 1.0 1.25}"
METRICS="${METRICS:-l1 mse psnr ssim temporal_l1}"
PRIMARY_METRIC="${PRIMARY_METRIC:-l1}"
LIMIT="${LIMIT:-10}"
SEED="${SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

[[ -f "${LORA_CKPT}" ]] || { echo "LoRA checkpoint not found: ${LORA_CKPT}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}"

strength_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "${value}"
}

link_case() {
  local source_root="$1" destination_root="$2" case_name="$3" source
  mkdir -p "${destination_root}/videos/${case_name}"
  for source in "${source_root}/videos/${case_name}"/*.mp4; do
    [[ -f "${source}" ]] || continue
    ln -f "${source}" "${destination_root}/videos/${case_name}/$(basename "${source}")"
  done
}

shared_root="${OUT_ROOT}/_shared_baselines"
if [[ ! -f "${shared_root}/videos/original3_clean_pred/original3_clean_pred_00_seed${SEED}.mp4" || \
      ! -f "${shared_root}/videos/teacher4/teacher4_00_seed${SEED}.mp4" ]]; then
  echo "[strength] generating shared original3 and teacher4 baselines"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LORA_CKPT="${LORA_CKPT}" CASES="original teacher" CREATE_COMPARE=0 \
  OUT_ROOT="${shared_root}" LIMIT="${LIMIT}" SEED="${SEED}" \
  bash "${SCRIPT_DIR}/run_last_step_skip_lora_clean_pred_compare.sh"
else
  echo "[strength] reuse shared baselines: ${shared_root}"
fi

for strength in ${STRENGTHS}; do
  tag="$(strength_tag "${strength}")"
  run_root="${OUT_ROOT}/strength_${tag}"
  summary="${run_root}/metrics/original_lora_teacher_summary.csv"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${summary}" ]]; then
    echo "[strength] reuse completed strength=${strength}"
    continue
  fi
  link_case "${shared_root}" "${run_root}" original3_clean_pred
  link_case "${shared_root}" "${run_root}" teacher4
  echo "[strength] generating LoRA strength=${strength}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LORA_CKPT="${LORA_CKPT}" LORA_STRENGTH="${strength}" CASES="lora" CREATE_COMPARE=0 \
  OUT_ROOT="${run_root}" LIMIT="${LIMIT}" SEED="${SEED}" \
  bash "${SCRIPT_DIR}/run_last_step_skip_lora_clean_pred_compare.sh"
  OUT_ROOT="${run_root}" \
  bash "${SCRIPT_DIR}/run_last_step_skip_lora_video_closeness_eval.sh" \
    --metrics ${METRICS} --primary_metric "${PRIMARY_METRIC}"
done

python - "${OUT_ROOT}" "${LORA_CKPT}" "${PRIMARY_METRIC}" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2]).resolve()
primary = sys.argv[3]
rows = []
rank_rows = []
for summary in sorted(root.glob("strength_*/metrics/original_lora_teacher_summary.csv")):
    strength = summary.parent.parent.name.removeprefix("strength_").replace("p", ".").replace("m", "-")
    with summary.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row = {"strength": strength, "lora_checkpoint": str(checkpoint), **source, "summary_csv": str(summary)}
            rows.append(row)
            if source.get("metric") == primary:
                rank_rows.append(row)

if not rows:
    raise SystemExit("No strength summaries found")

aggregate = root / "strength_metric_summary.csv"
with aggregate.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def score(row):
    value = float(row["lora_mean"])
    return value if row.get("better") == "lower" else -value

rank_rows.sort(key=score)
ranking = root / f"strength_rank_by_{primary}.csv"
with ranking.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0]))
    writer.writeheader()
    writer.writerows(rank_rows)

digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
stat = checkpoint.stat()
manifest = {
    "checkpoint": str(checkpoint),
    "checkpoint_size_bytes": stat.st_size,
    "checkpoint_mtime_ns": stat.st_mtime_ns,
    "checkpoint_sha256": digest.hexdigest(),
    "primary_metric": primary,
    "ranking": str(ranking),
}
(root / "strength_selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"aggregate={aggregate}")
print(f"ranking={ranking}")
PY

echo "Strength selection ready: ${OUT_ROOT}/strength_rank_by_${PRIMARY_METRIC}.csv"
