#!/usr/bin/env bash
set -euo pipefail

# Evaluate whether a LoRA trained on the 480p four-step trajectory transfers to
# the 368x640 handoff. Base3/teacher4 videos are generated once and hard-linked
# into subsequent strength runs; only the LoRA case is regenerated per strength.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LORA_CKPT="${LORA_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/latest.safetensors}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/aaai27_experiments/distill_480p_lora_transfer_368p}"
STRENGTHS="${STRENGTHS:-0.25 0.5 0.75 1.0}"
LIMIT="${LIMIT:-10}"
SEED="${SEED:-9800}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for path in "${LORA_CKPT}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}"; do
  [[ -f "${path}" ]] || { echo "File not found: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_ROOT}"
shared_root=""
run_roots=()

strength_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "${value}"
}

reuse_baselines() {
  local source_root="$1" destination_root="$2" case_name source
  for case_name in base3_stage2_hr4 teacher4_interp teacher4_stage2; do
    mkdir -p "${destination_root}/videos/${case_name}"
    for source in "${source_root}/videos/${case_name}"/*.mp4; do
      [[ -f "${source}" ]] || continue
      ln -f "${source}" "${destination_root}/videos/${case_name}/$(basename "${source}")"
    done
  done
}

for strength in ${STRENGTHS}; do
  tag="$(strength_tag "${strength}")"
  run_root="${OUT_ROOT}/strength_${tag}"
  run_roots+=("${run_root}")
  if [[ -n "${shared_root}" ]]; then
    reuse_baselines "${shared_root}" "${run_root}"
  fi

  echo "[transfer] strength=${strength} out=${run_root}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LORA_CKPT="${LORA_CKPT}" \
  LORA_STRENGTH="${strength}" \
  STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT}" \
  STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG}" \
  OUT_ROOT="${run_root}" \
  LIMIT="${LIMIT}" \
  SEED="${SEED}" \
  SKIP_EXISTING=1 \
  bash "${SCRIPT_DIR}/run_distill_368x640_720x1248_stage2_four_way.sh" run

  if [[ -z "${shared_root}" ]]; then
    shared_root="${run_root}"
  fi
done

python - "${OUT_ROOT}" "${LORA_CKPT}" "${run_roots[@]}" <<'PY'
import csv
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
lora_ckpt = sys.argv[2]
run_roots = [Path(value) for value in sys.argv[3:]]
rows = []
for run_root in run_roots:
    summary = run_root / "evaluation/distill_360p_summary.csv"
    if not summary.is_file():
        continue
    strength = run_root.name.removeprefix("strength_").replace("p", ".").replace("m", "-")
    with summary.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"strength": strength, "lora_checkpoint": lora_ckpt, **row})

if not rows:
    raise SystemExit("No transfer-sweep summaries found")
output = out_root / "transfer_sweep_summary.csv"
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Transfer summary: {output}")
PY

echo "Transfer sweep ready: ${OUT_ROOT}/transfer_sweep_summary.csv"
