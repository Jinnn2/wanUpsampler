#!/usr/bin/env bash
set -euo pipefail

# Sweep a handoff-step LoRA strength at the valid 360p-class resolution 368x640.
# Shared cases are generated once:
#   ori_${TRAIN_STEP} -> LoRA strengths -> ori_${INFER_STEPS}
# Each LoRA case is evaluated against the full teacher endpoint with the
# unmodified handoff prediction as the baseline.

MODE="${1:-run}"
if [[ "${MODE}" != "run" && "${MODE}" != "check" ]]; then
  echo "Usage: $0 [run|check]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/path/to/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/Wan-AI/Wan2.1-T2V-1.3B}"
TRAIN_STEP="${TRAIN_STEP:-45}"
INFER_STEPS="${INFER_STEPS:-50}"
TAIL_SKIP_LORA_OUT_DIR="${TAIL_SKIP_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step${TRAIN_STEP}_to_step${INFER_STEPS}}"
LORA_CKPT="${LORA_CKPT:-${TAIL_SKIP_LORA_OUT_DIR}/latest.safetensors}"

HEIGHT="${HEIGHT:-368}"
WIDTH="${WIDTH:-640}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
PRECISION="${PRECISION:-bf16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STRENGTHS="${STRENGTHS:-0.5 0.75 1.0}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
if [[ -z "${OUT_ROOT:-}" ]]; then
  if [[ "${TRAIN_STEP}" == "45" && "${INFER_STEPS}" == "50" ]]; then
    # Preserve the canonical path used by the already-frozen step45 results.
    OUT_ROOT="${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_strength_sweep_360p_368x640"
  else
    OUT_ROOT="${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_strength_sweep_step${TRAIN_STEP}_360p_368x640"
  fi
fi
SEED="${SEED:-9700}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

RUN_METRICS="${RUN_METRICS:-1}"
METRICS="${METRICS:-l1 mse psnr temporal_l1}"
PRIMARY_METRIC="${PRIMARY_METRIC:-psnr}"
METRICS_CPU="${METRICS_CPU:-0}"

if [[ "${HEIGHT}" != "368" || "${WIDTH}" != "640" ]]; then
  echo "This entrypoint is locked to the 360p-class test size 368x640." >&2
  echo "Received HEIGHT=${HEIGHT}, WIDTH=${WIDTH}." >&2
  exit 2
fi
if (( TRAIN_STEP < 1 || TRAIN_STEP >= INFER_STEPS )); then
  echo "Invalid TRAIN_STEP=${TRAIN_STEP}; expected [1, $((INFER_STEPS - 1))]." >&2
  exit 2
fi

case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16." >&2
    exit 2
    ;;
esac

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
done
for path in "${LORA_CKPT}" "${PROMPTS_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

read -r -a strength_values <<<"${STRENGTHS}"
if [[ "${#strength_values[@]}" -eq 0 ]]; then
  echo "STRENGTHS must contain at least one numeric value." >&2
  exit 2
fi

strength_tag() {
  python - "$1" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise SystemExit("LoRA strength must be a finite non-negative number")
print(format(value, ".8g").replace("-", "m").replace(".", "p"))
PY
}

strength_cases=()
strength_pairs=()
BASE_CASE="ori_${TRAIN_STEP}"
TEACHER_CASE="ori_${INFER_STEPS}"
for strength in "${strength_values[@]}"; do
  tag="$(strength_tag "${strength}")"
  case_name="lora_${TRAIN_STEP}_s${tag}"
  strength_cases+=("${case_name}")
  strength_pairs+=("${strength}:${case_name}")
done

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos" "${OUT_ROOT}/compare" "${OUT_ROOT}/metrics"
mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

export LORA_CKPT TRAIN_STEP INFER_STEPS HEIGHT WIDTH NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT TEACHER_CASE

write_config() {
  local output="$1"
  local case_name="$2"
  local strength="$3"
  python - "${output}" "${case_name}" "${strength}" <<'PY'
import json
import os
import sys

path, case_name, strength = sys.argv[1:]
train_step = int(os.environ["TRAIN_STEP"])
config = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": int(os.environ["HEIGHT"]),
    "target_width": int(os.environ["WIDTH"]),
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": float(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "compare_name": case_name,
}

if case_name != os.environ["TEACHER_CASE"]:
    config.update(
        {
            "lora_dynamic_apply": True,
            "lora_active_steps": [train_step],
            "return_clean_pred_steps": [train_step],
            "lora_configs": [
                {
                    "name": "wan2.1",
                    "path": os.environ["LORA_CKPT"],
                    "strength": float(strength),
                }
            ],
        }
    )

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

write_config "${OUT_ROOT}/configs/${BASE_CASE}.json" "${BASE_CASE}" "0"
write_config "${OUT_ROOT}/configs/${TEACHER_CASE}.json" "${TEACHER_CASE}" "0"
for index in "${!strength_values[@]}"; do
  write_config \
    "${OUT_ROOT}/configs/${strength_cases[${index}]}.json" \
    "${strength_cases[${index}]}" \
    "${strength_values[${index}]}"
done

column_names=("${BASE_CASE}" "${strength_cases[@]}" "${TEACHER_CASE}")
column_text="$(IFS=' | '; echo "${column_names[*]}")"
echo "[360p strength sweep] prompts=${PROMPTS_FILE} selected=${#prompts[@]} seed=${SEED}"
echo "[360p strength sweep] resolution=${HEIGHT}x${WIDTH} (latent 46x80), step=${TRAIN_STEP}"
echo "[360p strength sweep] strengths=${STRENGTHS}"
echo "[360p strength sweep] columns=${column_text}"

if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; configs written under ${OUT_ROOT}/configs and inference was not started."
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

run_infer() {
  local model_cls="$1"
  local case_name="$2"
  local sample_index="$3"
  local prompt="$4"
  local sample_seed="$5"
  local output_dir="${OUT_ROOT}/videos/${case_name}"
  local output="${output_dir}/${case_name}_${sample_index}_seed${sample_seed}.mp4"

  mkdir -p "${output_dir}"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${output}" ]]; then
    echo "  [skip] ${case_name}: ${output}"
    return
  fi

  python "${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py" \
    --seed "${sample_seed}" \
    --model_cls "${model_cls}" \
    --task t2v \
    --model_path "${MODEL_ROOT}" \
    --config_json "${OUT_ROOT}/configs/${case_name}.json" \
    --prompt "${prompt}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${output}" \
    --target_video_length "${NUM_FRAMES}"
}

make_compare() {
  local sample_index="$1"
  local sample_seed="$2"
  local out="${OUT_ROOT}/compare/${sample_index}_seed${sample_seed}_strength_sweep_hstack.mp4"
  local inputs=()
  local filters=()
  local stack_inputs=""
  local input_index=0

  for case_name in "${column_names[@]}"; do
    video="${OUT_ROOT}/videos/${case_name}/${case_name}_${sample_index}_seed${sample_seed}.mp4"
    if [[ ! -f "${video}" ]]; then
      echo "[compare] skip hstack for index=${sample_index}; missing ${video}." >&2
      return
    fi
    inputs+=(-i "${video}")
    filters+=("[${input_index}:v]setpts=PTS-STARTPTS[v${input_index}]")
    stack_inputs+="[v${input_index}]"
    input_index=$((input_index + 1))
  done

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[compare] ffmpeg unavailable; individual videos were still written."
    return
  fi

  filter_prefix="$(IFS=';'; echo "${filters[*]}")"
  ffmpeg -hide_banner -loglevel error -y \
    "${inputs[@]}" \
    -filter_complex "${filter_prefix};${stack_inputs}hstack=inputs=${#column_names[@]}[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "${out}"
}

index=0
for prompt in "${prompts[@]}"; do
  sample_index=$((PROMPT_OFFSET + index))
  sample_seed=$((SEED + sample_index))
  sample_label="$(printf "%02d" "${sample_index}")"

  echo "[$((index + 1))/${#prompts[@]}] index=${sample_label} seed=${sample_seed}"
  echo "${prompt}"
  run_infer "wan2.1_tail_skip_lora" "${BASE_CASE}" "${sample_label}" "${prompt}" "${sample_seed}"
  for case_name in "${strength_cases[@]}"; do
    run_infer "wan2.1_tail_skip_lora" "${case_name}" "${sample_label}" "${prompt}" "${sample_seed}"
  done
  run_infer "wan2.1" "${TEACHER_CASE}" "${sample_label}" "${prompt}" "${sample_seed}"
  make_compare "${sample_label}" "${sample_seed}"
  index=$((index + 1))
done

if [[ "${RUN_METRICS}" == "1" ]]; then
  read -r -a metric_args <<<"${METRICS}"
  cpu_args=()
  if [[ "${METRICS_CPU}" == "1" ]]; then
    cpu_args+=(--cpu)
  fi

  for index in "${!strength_values[@]}"; do
    strength="${strength_values[${index}]}"
    case_name="${strength_cases[${index}]}"
    result_dir="${OUT_ROOT}/metrics/${case_name}"
    python "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/eval_last_step_skip_lora_video_closeness.py" \
      --out_root "${OUT_ROOT}" \
      --original_case "${BASE_CASE}" \
      --lora_case "${case_name}" \
      --teacher_case "${TEACHER_CASE}" \
      --result_dir "${result_dir}" \
      --metrics "${metric_args[@]}" \
      --primary_metric "${PRIMARY_METRIC}" \
      "${cpu_args[@]}"
  done

  python - "${OUT_ROOT}" "${strength_pairs[@]}" <<'PY'
import csv
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
pairs = [item.split(":", 1) for item in sys.argv[2:]]
rows = []
for strength, case_name in pairs:
    path = out_root / "metrics" / case_name / "original_lora_teacher_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"strength": strength, "case_name": case_name, **row})

output = out_root / "metrics" / "strength_sweep_summary.csv"
fieldnames = ["strength", "case_name", *[key for key in rows[0] if key not in {"strength", "case_name"}]]
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

for metric in sorted({row["metric"] for row in rows}):
    candidates = [row for row in rows if row["metric"] == metric]
    reverse = candidates[0]["better"] == "higher"
    best = sorted(candidates, key=lambda row: float(row["lora_mean"]), reverse=reverse)[0]
    print(
        f"[best] metric={metric} strength={best['strength']} "
        f"lora_mean={best['lora_mean']} win_rate={best['lora_win_rate']}"
    )
print(f"Combined strength summary: {output}")
PY
fi

echo "360p strength-sweep comparisons: ${OUT_ROOT}/compare"
echo "Combined metrics (if enabled): ${OUT_ROOT}/metrics/strength_sweep_summary.csv"
