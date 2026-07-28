#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/path/to/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/Wan-AI/Wan2.1-T2V-1.3B}"
PROMPTS_FILE="${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}"
OUT_DIR="${CR_TIME_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_time_compare}"

DIRECT_CONFIG="${DIRECT_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p.json}"
STAGE3_CHECKPOINT="${STAGE3_CHECKPOINT:-${CR_STAGE3_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_x0pred_480p720p_stage3_lmdb}/latest.pt}"
STAGE3_TRAIN_CONFIG="${STAGE3_TRAIN_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_x0pred_480p_to_720p_lmdb_stage3.yaml}"
X0PRED_SOURCE_LMDB="${X0PRED_SOURCE_LMDB:-${CR_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}}"
X0PRED_CONFIG="${X0PRED_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_x0pred_480p.json}"

BENCH_CASES="${BENCH_CASES:-direct_720p,stage3_bridge_720p}"
REPEATS="${REPEATS:-1}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9100}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
PRECISION="${PRECISION:-bf16}"

INFER_STEPS="${INFER_STEPS:-50}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
CHANGE_STEP="${CHANGE_STEP:-35}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
LR_H="${LR_H:-480}"
LR_W="${LR_W:-832}"
USE_EMA="${USE_EMA:-1}"
STAGE3_MODEL_CLASS="${STAGE3_MODEL_CLASS:-stage2}"
STAGE3_RESIDUAL_SKIP="${STAGE3_RESIDUAL_SKIP:-checkpoint}"

X0PRED_DENOISE_STEP="${X0PRED_DENOISE_STEP:-45}"
X0PRED_MAX_SAMPLES="${X0PRED_MAX_SAMPLES:-1}"
X0PRED_SAMPLE_OFFSET="${X0PRED_SAMPLE_OFFSET:-0}"

export CUDA_VISIBLE_DEVICES LIGHTX2V_REPO PROJECT_ROOT
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"
case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  fp32) export DTYPE="${DTYPE:-FP32}" ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16, fp16, or fp32" >&2
    exit 2
    ;;
esac

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Directory not found: ${path}" >&2
    exit 1
  fi
done
for path in "${PROMPTS_FILE}" "${DIRECT_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "File not found: ${path}" >&2
    exit 1
  fi
done

if [[ -f "${LIGHTX2V_REPO}/scripts/base/base.sh" ]]; then
  lightx2v_path="${LIGHTX2V_REPO}"
  model_path="${MODEL_ROOT}"
  # shellcheck source=/dev/null
  source "${LIGHTX2V_REPO}/scripts/base/base.sh"
fi

mkdir -p "${OUT_DIR}"/{configs,videos,x0pred,logs}
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_DIR}/time_summary.csv}"
echo "case,repeat,seed,elapsed_sec,output,log" >"${SUMMARY_CSV}"

mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${REPEATS}")
if [[ "${#prompts[@]}" -eq 0 ]]; then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

now() {
  python -c 'import time; print(f"{time.perf_counter():.9f}")'
}

elapsed() {
  python - "$1" "$2" <<'PY'
import sys
start = float(sys.argv[1])
end = float(sys.argv[2])
print(f"{end - start:.3f}")
PY
}

write_stage3_config() {
  local output="$1"
  python - "$output" <<'PY'
import json
import os
import sys

path = sys.argv[1]
use_ema = os.environ["USE_EMA"] == "1"
residual_skip = os.environ["STAGE3_RESIDUAL_SKIP"].lower()
if residual_skip not in {"checkpoint", "on", "off", "true", "false", "1", "0"}:
    raise SystemExit("STAGE3_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0")

cfg = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": int(os.environ["HR_H"]),
    "target_width": int(os.environ["HR_W"]),
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": float(os.environ["GUIDE_SCALE"]),
    "sample_shift": float(os.environ["SAMPLE_SHIFT"]),
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "changing_resolution": True,
    "resolution_rate": [int(os.environ["LR_H"]) / int(os.environ["HR_H"])],
    "changing_resolution_steps": [int(os.environ["CHANGE_STEP"])],
    "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
    "wan_clean_resizer_ckpt": os.environ["STAGE3_CHECKPOINT"],
    "wan_clean_resizer_train_config": os.environ["STAGE3_TRAIN_CONFIG"],
    "wan_clean_resizer_model_class": os.environ["STAGE3_MODEL_CLASS"],
    "wan_clean_resizer_use_ema": use_ema,
}
if residual_skip != "checkpoint":
    cfg["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
PY
}

append_summary() {
  local case_name="$1"
  local repeat_id="$2"
  local seed="$3"
  local elapsed_sec="$4"
  local output="$5"
  local log="$6"
  printf '%s,%s,%s,%s,%q,%q\n' "${case_name}" "${repeat_id}" "${seed}" "${elapsed_sec}" "${output}" "${log}" >>"${SUMMARY_CSV}"
}

run_timed() {
  local case_name="$1"
  local repeat_id="$2"
  local seed="$3"
  local output="$4"
  local log="$5"
  shift 5

  echo "===== ${case_name} repeat=${repeat_id} seed=${seed} ====="
  echo "output: ${output}"
  local start end seconds
  start="$(now)"
  "$@" 2>&1 | tee "${log}"
  end="$(now)"
  seconds="$(elapsed "${start}" "${end}")"
  echo "${case_name} repeat=${repeat_id} elapsed=${seconds}s"
  append_summary "${case_name}" "${repeat_id}" "${seed}" "${seconds}" "${output}" "${log}"
}

IFS=',' read -r -a cases <<< "${BENCH_CASES}"

for repeat_id in "${!prompts[@]}"; do
  prompt="${prompts[repeat_id]}"
  seed=$((START_SEED + PROMPT_OFFSET + repeat_id))
  sample_id="$(printf "%03d_seed%s" "$((PROMPT_OFFSET + repeat_id))" "${seed}")"

  for case_name in "${cases[@]}"; do
    case_name="$(echo "${case_name}" | xargs)"
    [[ -z "${case_name}" ]] && continue

    if [[ "${case_name}" == "direct_720p" ]]; then
      output="${OUT_DIR}/videos/${sample_id}_direct_720p.mp4"
      log="${OUT_DIR}/logs/${sample_id}_direct_720p.log"
      run_timed "${case_name}" "${repeat_id}" "${seed}" "${output}" "${log}" \
        python -m lightx2v.infer \
          --seed "${seed}" \
          --model_cls wan2.1 \
          --task t2v \
          --model_path "${MODEL_ROOT}" \
          --config_json "${DIRECT_CONFIG}" \
          --prompt "${prompt}" \
          --negative_prompt "${NEGATIVE_PROMPT}" \
          --save_result_path "${output}"
    elif [[ "${case_name}" == "stage3_bridge_720p" ]]; then
      for path in "${STAGE3_CHECKPOINT}" "${STAGE3_TRAIN_CONFIG}"; do
        if [[ ! -f "${path}" ]]; then
          echo "File not found for stage3_bridge_720p: ${path}" >&2
          exit 1
        fi
      done
      config_json="${OUT_DIR}/configs/${sample_id}_stage3_bridge_720p.json"
      output="${OUT_DIR}/videos/${sample_id}_stage3_bridge_720p.mp4"
      log="${OUT_DIR}/logs/${sample_id}_stage3_bridge_720p.log"
      export STAGE3_CHECKPOINT STAGE3_TRAIN_CONFIG USE_EMA STAGE3_MODEL_CLASS STAGE3_RESIDUAL_SKIP
      export INFER_STEPS NUM_FRAMES HR_H HR_W LR_H LR_W GUIDE_SCALE SAMPLE_SHIFT CHANGE_STEP
      write_stage3_config "${config_json}"
      run_timed "${case_name}" "${repeat_id}" "${seed}" "${output}" "${log}" \
        python "${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py" \
          --seed "${seed}" \
          --model_cls wan2.1_clean_resizer_bridge \
          --task t2v \
          --model_path "${MODEL_ROOT}" \
          --config_json "${config_json}" \
          --prompt "${prompt}" \
          --negative_prompt "${NEGATIVE_PROMPT}" \
          --save_result_path "${output}"
    elif [[ "${case_name}" == "x0pred_call" ]]; then
      if [[ ! -d "${X0PRED_SOURCE_LMDB}" ]]; then
        echo "X0PRED_SOURCE_LMDB not found: ${X0PRED_SOURCE_LMDB}" >&2
        exit 1
      fi
      output="${OUT_DIR}/x0pred/${sample_id}_step${X0PRED_DENOISE_STEP}"
      log="${OUT_DIR}/logs/${sample_id}_x0pred_call_step${X0PRED_DENOISE_STEP}.log"
      rm -rf "${output}"
      run_timed "${case_name}" "${repeat_id}" "${seed}" "${output}" "${log}" \
        python "${PROJECT_ROOT}/changing_resolution/scripts/data/build_x0pred_480p720p_stage3_lmdb.py" \
          --source_lmdb "${X0PRED_SOURCE_LMDB}" \
          --out_dir "${output}" \
          --mode lightx2v \
          --lightx2v_repo "${LIGHTX2V_REPO}" \
          --model_path "${MODEL_ROOT}" \
          --config_json "${X0PRED_CONFIG}" \
          --infer_steps "${INFER_STEPS}" \
          --denoise_step "${X0PRED_DENOISE_STEP}" \
          --sample_shift "${SAMPLE_SHIFT}" \
          --sample_guide_scale "${GUIDE_SCALE}" \
          --base_seed "${seed}" \
          --offset "${X0PRED_SAMPLE_OFFSET}" \
          --max_samples "${X0PRED_MAX_SAMPLES}" \
          --precision "${PRECISION}" \
          --overwrite
    else
      echo "Unknown case: ${case_name}" >&2
      echo "Supported cases: direct_720p,stage3_bridge_720p,x0pred_call" >&2
      exit 2
    fi
  done
done

echo "Timing summary: ${SUMMARY_CSV}"
