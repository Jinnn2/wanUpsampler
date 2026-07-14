#!/usr/bin/env bash
set -euo pipefail

# Sweep the LR handoff step at the valid Wan 360p-class size (368x640).
# For every step N, the three output columns are:
#   1. LR denoise step N -> new 360p Stage2 -> HR denoise steps N+1..50
#   2. LR denoise step N -> trilinear resize -> HR denoise steps N+1..50
#   3. LR denoise through step 50 -> new 360p Stage2 -> decode (fixed baseline)

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

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_STAGE2_368X640_720X1248_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${CR_STAGE2_368X640_720X1248_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/changing_resolution_clean_360p_stage2_three_way_step_sweep}"

STEP_START="${STEP_START:-30}"
STEP_END="${STEP_END:-50}"
STEP_STRIDE="${STEP_STRIDE:-1}"
CHANGE_STEPS="${CHANGE_STEPS:-}"
INFER_STEPS="${INFER_STEPS:-50}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
START_SEED="${START_SEED:-9700}"

LR_H=368
LR_W=640
HR_H=720
HR_W=1248
LR_LATENT_H=46
LR_LATENT_W=80
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-16}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-8}"
PRECISION="${PRECISION:-bf16}"
USE_EMA="${USE_EMA:-0}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

if (( INFER_STEPS != 50 )); then
  echo "This comparison is defined against a 50-step baseline; INFER_STEPS must be 50." >&2
  exit 2
fi
if (( STEP_STRIDE < 1 )); then
  echo "STEP_STRIDE must be at least 1." >&2
  exit 2
fi

if [[ -n "${CHANGE_STEPS}" ]]; then
  read -r -a steps <<< "${CHANGE_STEPS}"
else
  steps=()
  step="${STEP_START}"
  while (( step <= STEP_END )); do
    steps+=("${step}")
    step=$((step + STEP_STRIDE))
  done
fi
if (( ${#steps[@]} == 0 )); then
  echo "No change steps selected." >&2
  exit 2
fi
for step in "${steps[@]}"; do
  if [[ ! "${step}" =~ ^[0-9]+$ ]] || (( step < 1 || step > INFER_STEPS )); then
    echo "Invalid change step ${step}; expected an integer in [1, ${INFER_STEPS}]." >&2
    exit 2
  fi
done

case "${PRECISION}" in
  bf16) export DTYPE="${DTYPE:-BF16}" ;;
  fp16) export DTYPE="${DTYPE:-FP16}" ;;
  *) echo "Unsupported PRECISION=${PRECISION}; use bf16 or fp16." >&2; exit 2 ;;
esac

for path in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  [[ -d "${path}" ]] || { echo "Directory not found: ${path}" >&2; exit 1; }
done
for path in "${PROMPTS_FILE}" "${STAGE2_CHECKPOINT}" "${STAGE2_TRAIN_CONFIG}"; do
  [[ -f "${path}" ]] || { echo "File not found: ${path}" >&2; exit 1; }
done
if [[ "${MODE}" == "run" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to build the labeled three-column videos." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"/{configs,videos/stage2_handoff,videos/interp_handoff,videos/baseline50_stage2,panels,compare}
mapfile -t prompts < <(grep -v '^[[:space:]]*$' "${PROMPTS_FILE}" | grep -v '^[[:space:]]*#' | tail -n +"$((PROMPT_OFFSET + 1))" | head -n "${LIMIT}")
if (( ${#prompts[@]} == 0 )); then
  echo "No prompts selected from: ${PROMPTS_FILE}" >&2
  exit 1
fi

BRIDGE_USE_EMA=false
if [[ "${USE_EMA}" == "1" ]]; then
  BRIDGE_USE_EMA=true
elif [[ "${USE_EMA}" != "0" ]]; then
  echo "USE_EMA must be 0 or 1." >&2
  exit 2
fi

export PROJECT_ROOT STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG INFER_STEPS NUM_FRAMES
export GUIDE_SCALE SAMPLE_SHIFT BRIDGE_USE_EMA STAGE2_RESIDUAL_SKIP
export LR_H LR_W HR_H HR_W LR_LATENT_H LR_LATENT_W

write_config() {
  local output="$1"
  local mode="$2"
  local change_step="$3"
  python - "${output}" "${mode}" "${change_step}" <<'PY'
import json
import os
import sys

path, mode, change_step = sys.argv[1], sys.argv[2], int(sys.argv[3])
if mode not in {"stage2", "interp"}:
    raise SystemExit(f"Unsupported mode: {mode}")

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
    "wan_lowres_latent_size": [
        int(os.environ["LR_LATENT_H"]),
        int(os.environ["LR_LATENT_W"]),
    ],
    "changing_resolution_steps": [change_step],
}

if mode == "stage2":
    cfg.update(
        {
            "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
            "wan_clean_resizer_ckpt": os.environ["STAGE2_CHECKPOINT"],
            "wan_clean_resizer_train_config": os.environ["STAGE2_TRAIN_CONFIG"],
            "wan_clean_resizer_model_class": "stage2",
            "wan_clean_resizer_use_ema": os.environ["BRIDGE_USE_EMA"].lower() == "true",
        }
    )
    residual_skip = os.environ["STAGE2_RESIDUAL_SKIP"].lower()
    valid = {"checkpoint", "on", "off", "true", "false", "1", "0"}
    if residual_skip not in valid:
        raise SystemExit(
            "STAGE2_RESIDUAL_SKIP must be checkpoint, on/off, true/false, or 1/0"
        )
    if residual_skip != "checkpoint":
        cfg["wan_clean_resizer_residual_skip"] = residual_skip in {"on", "true", "1"}

with open(path, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

# Configs do not depend on prompt. The step-50 Stage2 config is also the fixed
# baseline and is intentionally shared by every row of the sweep.
for change_step in "${steps[@]}"; do
  write_config "${OUT_ROOT}/configs/step$(printf '%02d' "${change_step}")_stage2.json" "stage2" "${change_step}"
  write_config "${OUT_ROOT}/configs/step$(printf '%02d' "${change_step}")_interp.json" "interp" "${change_step}"
done
write_config "${OUT_ROOT}/configs/baseline50_stage2.json" "stage2" "${INFER_STEPS}"

echo "[360p three-way sweep] prompts=${#prompts[@]}, steps=${steps[*]}"
echo "[360p three-way sweep] LR=368x640 (latent 46x80), HR=720x1248 (latent 90x156)"
echo "[360p three-way sweep] Stage2=${STAGE2_CHECKPOINT}"
echo "[360p three-way sweep] columns: step N + Stage2 + HR tail | step N + interp + HR tail | LR50 + Stage2"

if [[ "${MODE}" == "check" ]]; then
  echo "Check passed; configs written under ${OUT_ROOT}/configs and inference was not started."
  exit 0
fi

export CUDA_VISIBLE_DEVICES LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

run_infer() {
  local model_cls="$1"
  local config_json="$2"
  local prompt="$3"
  local seed="$4"
  local output="$5"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${output}" ]]; then
    echo "  [skip] ${output}"
    return
  fi
  python "${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py" \
    --seed "${seed}" \
    --model_cls "${model_cls}" \
    --task t2v \
    --model_path "${MODEL_ROOT}" \
    --config_json "${config_json}" \
    --prompt "${prompt}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --save_result_path "${output}" \
    --target_video_length "${NUM_FRAMES}"
}

make_panel() {
  local input="$1"
  local output="$2"
  local label="$3"
  ffmpeg -hide_banner -loglevel error -y -i "${input}" \
    -vf "scale=${HR_W}:${HR_H}:flags=bicubic,fps=${FPS},drawbox=x=0:y=0:w=iw:h=46:color=black@0.55:t=fill,drawtext=text='${label}':x=20:y=12:fontsize=24:fontcolor=white" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${output}"
}

index=0
for prompt in "${prompts[@]}"; do
  global_index=$((PROMPT_OFFSET + index))
  seed=$((START_SEED + global_index))
  sample_label="$(printf '%03d' "${global_index}")"
  baseline="${OUT_ROOT}/videos/baseline50_stage2/${sample_label}_seed${seed}_baseline50_stage2.mp4"
  baseline_panel="${OUT_ROOT}/panels/${sample_label}_seed${seed}_baseline50_stage2.mp4"

  echo "[$((index + 1))/${#prompts[@]}] index=${sample_label} seed=${seed}: fixed LR50+Stage2 baseline"
  echo "${prompt}"
  run_infer "wan2.1_clean_resizer_bridge" "${OUT_ROOT}/configs/baseline50_stage2.json" "${prompt}" "${seed}" "${baseline}"
  make_panel "${baseline}" "${baseline_panel}" "LR step 50 + Stage2 baseline"

  for change_step in "${steps[@]}"; do
    step_label="$(printf '%02d' "${change_step}")"
    stem="${sample_label}_seed${seed}_step${step_label}"
    stage2_video="${OUT_ROOT}/videos/stage2_handoff/${stem}_stage2.mp4"
    interp_video="${OUT_ROOT}/videos/interp_handoff/${stem}_interp.mp4"
    stage2_panel="${OUT_ROOT}/panels/${stem}_stage2.mp4"
    interp_panel="${OUT_ROOT}/panels/${stem}_interp.mp4"
    compare="${OUT_ROOT}/compare/${stem}_three_way.mp4"

    echo "  step=${change_step}/${INFER_STEPS}"
    if (( change_step == INFER_STEPS )); then
      # This is exactly the fixed baseline; avoid a duplicate model run.
      stage2_video="${baseline}"
      make_panel "${stage2_video}" "${stage2_panel}" "LR step 50 + Stage2 (same as baseline)"
    else
      run_infer "wan2.1_clean_resizer_bridge" "${OUT_ROOT}/configs/step${step_label}_stage2.json" "${prompt}" "${seed}" "${stage2_video}"
      make_panel "${stage2_video}" "${stage2_panel}" "LR step ${change_step} + Stage2 + HR to 50"
    fi
    run_infer "wan2.1_clean_interp_bridge" "${OUT_ROOT}/configs/step${step_label}_interp.json" "${prompt}" "${seed}" "${interp_video}"
    make_panel "${interp_video}" "${interp_panel}" "LR step ${change_step} + interp + HR to 50"

    ffmpeg -hide_banner -loglevel error -y \
      -i "${stage2_panel}" -i "${interp_panel}" -i "${baseline_panel}" \
      -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
      -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 18 "${compare}"
  done
  index=$((index + 1))
done

echo "360p three-way step-sweep videos: ${OUT_ROOT}/compare"
