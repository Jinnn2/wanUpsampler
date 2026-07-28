#!/usr/bin/env bash
set -euo pipefail

# Distill-only 368x640 -> 720x1248 evaluation:
#   base3_stage2_hr4 | lora3_stage2_hr4 | teacher4_interp | teacher4_stage2

MODE="${1:-run}"
if [[ "${MODE}" != "check" && "${MODE}" != "run" && "${MODE}" != "eval" ]]; then
  echo "Usage: $0 [check|run|eval]" >&2
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
MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-/path/to/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${MODEL_ROOT}/distill_model.pt}"
LORA_CKPT="${LORA_CKPT:-${CR_DISTILL_360_LORA_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3}/latest.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_DISTILL_360_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_DISTILL_360_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml}}"

PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_stage2_four_way_368x640_720x1248}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-9800}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5}"
RENOISE_MODE="${RENOISE_MODE:-random}"
STAGE2_USE_EMA="${STAGE2_USE_EMA:-1}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *) echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16." >&2; exit 2 ;;
esac

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  [[ -d "${path}" ]] || { echo "Directory not found: ${path}" >&2; exit 1; }
done
for path in "${DIT_CKPT}" "${LORA_CKPT}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}" "${PROMPTS_FILE}"; do
  [[ -f "${path}" ]] || { echo "File not found: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/videos" "${OUT_ROOT}/compare" "${OUT_ROOT}/evaluation"
mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
(( ${#prompts[@]} > 0 )) || { echo "No prompts selected from: ${PROMPTS_FILE}" >&2; exit 1; }

export PROJECT_ROOT DIT_CKPT LORA_CKPT LORA_STRENGTH STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG
export NUM_FRAMES GUIDE_SCALE SAMPLE_SHIFT RENOISE_MODE STAGE2_USE_EMA STAGE2_RESIDUAL_SKIP

write_config() {
  local output="$1"
  local case_name="$2"
  python - "${output}" "${case_name}" <<'PY'
import json
import os
import sys

path, case_name = sys.argv[1:]
valid = {"base3_stage2_hr4", "lora3_stage2_hr4", "teacher4_interp", "teacher4_stage2"}
if case_name not in valid:
    raise SystemExit(f"Unsupported case: {case_name}")

change_step = 3 if case_name in {"base3_stage2_hr4", "lora3_stage2_hr4"} else 4
config = {
    "infer_steps": 4,
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": 720,
    "target_width": 1248,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": float(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": False,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "denoising_step_list": [1000, 750, 500, 250],
    "dit_original_ckpt": os.environ["DIT_CKPT"],
    "changing_resolution": True,
    "resolution_rate": [368 / 720],
    "wan_lowres_latent_size": [46, 80],
    "changing_resolution_steps": [change_step],
    "wan_distill_bridge_renoise_mode": os.environ["RENOISE_MODE"],
    "compare_name": case_name,
}

if case_name != "teacher4_interp":
    config.update(
        {
            "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
            "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
            "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
            "wan_clean_resizer_model_class": "stage2",
            "wan_clean_resizer_use_ema": os.environ["STAGE2_USE_EMA"] == "1",
        }
    )
    residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
    if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
        raise SystemExit("Invalid STAGE2_RESIDUAL_SKIP")
    if residual_skip != "checkpoint":
        config["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

if case_name == "lora3_stage2_hr4":
    config.update(
        {
            "lora_dynamic_apply": True,
            "lora_active_steps": [3],
            "lora_configs": [
                {
                    "name": "wan2.1",
                    "path": os.environ["LORA_CKPT"],
                    "strength": float(os.environ["LORA_STRENGTH"]),
                }
            ],
        }
    )

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

for case_name in base3_stage2_hr4 lora3_stage2_hr4 teacher4_interp teacher4_stage2; do
  write_config "${OUT_ROOT}/configs/${case_name}.json" "${case_name}"
done

echo "[distill 360p] LR=368x640 (46x80) -> HR=720x1248 (90x156)"
echo "[distill 360p] cases: base3+Stage2+HR4 | LoRA3+Stage2+HR4 | teacher4+interp | teacher4+Stage2"
if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; configs written to ${OUT_ROOT}/configs"
  exit 0
fi

export CUDA_VISIBLE_DEVICES LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

run_case_batch() {
  local case_name="$1"
  local model_cls
  case "${case_name}" in
    lora3_stage2_hr4) model_cls=wan2.1_distill_last_step_lora_clean_resizer_bridge ;;
    teacher4_interp) model_cls=wan2.1_distill_interp_bridge ;;
    *) model_cls=wan2.1_distill_clean_resizer_bridge ;;
  esac
  local output_dir="${OUT_ROOT}/videos/${case_name}"
  mkdir -p "${output_dir}"
  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi
  python "${PROJECT_ROOT}/changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_batch_infer.py" \
    --seed "${SEED}" --increment_seed \
    --model_cls "${model_cls}" --task t2v --model_path "${MODEL_ROOT}" \
    --config_json "${OUT_ROOT}/configs/${case_name}.json" \
    --negative_prompt "${NEGATIVE_PROMPT}" --target_video_length "${NUM_FRAMES}" \
    --prompts_file "${PROMPTS_FILE}" --out_dir "${output_dir}" --name_prefix "${case_name}" \
    --prompt-offset "${PROMPT_OFFSET}" --limit "${LIMIT}" "${skip_args[@]}"
}

make_compare() {
  local index="$1" seed="$2"
  local out="${OUT_ROOT}/compare/${index}_seed${seed}_four_way.mp4"
  command -v ffmpeg >/dev/null 2>&1 || return
  local inputs=() filters=() labels=() i=0 video
  for case_name in base3_stage2_hr4 lora3_stage2_hr4 teacher4_interp teacher4_stage2; do
    video="${OUT_ROOT}/videos/${case_name}/${case_name}_${index}_seed${seed}.mp4"
    [[ -f "${video}" ]] || return
    inputs+=(-i "${video}")
    filters+=("[${i}:v]setpts=PTS-STARTPTS[v${i}]")
    labels+=("[v${i}]")
    i=$((i + 1))
  done
  local filter_chain
  filter_chain="$(IFS=';'; echo "${filters[*]}");${labels[*]}hstack=inputs=4[v]"
  ffmpeg -hide_banner -loglevel error -y "${inputs[@]}" -filter_complex "${filter_chain}" \
    -map '[v]' -an -c:v libx264 -pix_fmt yuv420p "${out}"
}

if [[ "${MODE}" == "run" ]]; then
  for case_name in base3_stage2_hr4 lora3_stage2_hr4 teacher4_interp teacher4_stage2; do
    echo "[batch] case=${case_name}; model weights load once for ${#prompts[@]} prompt(s)"
    run_case_batch "${case_name}"
  done
  for ((i=0; i<${#prompts[@]}; i++)); do
    sample_index=$((PROMPT_OFFSET + i))
    sample_label="$(printf '%02d' "${sample_index}")"
    sample_seed=$((SEED + sample_index))
    make_compare "${sample_label}" "${sample_seed}"
  done
fi

python "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/eval_last_step_skip_lora_video_closeness.py" \
  --out_root "${OUT_ROOT}" --result_dir "${OUT_ROOT}/evaluation" \
  --original_case base3_stage2_hr4 --lora_case lora3_stage2_hr4 --teacher_case teacher4_stage2 \
  --metrics lpips temporal_l1 psnr ssim l1 mse --primary_metric lpips \
  --limit "${LIMIT}" --jsonl_name distill_360p_metrics.jsonl \
  --csv_name distill_360p_metrics.csv --summary_json_name distill_360p_summary.json \
  --summary_csv_name distill_360p_summary.csv

echo "Videos:     ${OUT_ROOT}/videos"
echo "Comparisons:${OUT_ROOT}/compare"
echo "Metrics:    ${OUT_ROOT}/evaluation"
