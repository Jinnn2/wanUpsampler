#!/usr/bin/env bash
set -euo pipefail

# Final Distill4 quality suite:
#   1 native HR + 5 early handoffs + 3 endpoint domains x 4 HR budgets = 18 cases.
# Each case is ordinary single-GPU inference; cases are packed across four GPUs.

MODE="${1:-run}"
if [[ "${MODE}" != "check" && "${MODE}" != "run" ]]; then
  echo "Usage: $0 [check|run]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi
cd "${PROJECT_ROOT}"

WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/path/to/LightX2V}"
DIFFSYNTH_REPO="${DIFFSYNTH_REPO:-/path/to/DiffSynth-Studio}"
REALESRGAN_REPO="${REALESRGAN_REPO:-/path/to/Real-ESRGAN}"

MODEL_ROOT="${DISTILL_MODEL_ROOT:-/path/to/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
DIT_CKPT="${DISTILL_DIT_CKPT:-${MODEL_ROOT}/distill_model.pt}"
STAGE2_CHECKPOINT="${DISTILL_STAGE2_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb/latest.pt}"
STAGE2_TRAIN_CONFIG="${DISTILL_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml}"
LORA_CHECKPOINT="${DISTILL_LORA_480_CKPT:-${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/step_0010000.safetensors}"
REALESRGAN_X2_CHECKPOINT="${REALESRGAN_X2_CKPT:-/path/to/Real-ESRGAN/weights/RealESRGAN_x2plus.pth}"
REALESRGAN_X2_URL="${REALESRGAN_X2_URL:-}"
REALESRGAN_X2_BYTES="${REALESRGAN_X2_BYTES:-67061725}"
AUTO_DOWNLOAD_REALESRGAN="${AUTO_DOWNLOAD_REALESRGAN:-0}"
PROMPTS_FILE="${AAAI_PROMPTS:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${DISTILL4_FINAL_QUALITY_EFFICIENCY:-${PROJECT_ROOT}/outputs/aaai27_experiments/quality_efficiency_distill4}"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 4 )); then
  echo "Exactly four comma-separated GPU ids are required; got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "GPU ids must be non-negative integers; got ${gpu@Q}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU ids must be unique; duplicate ${gpu}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu}"]=1
done

SEED="${SEED:-9800}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
NUM_FRAMES="${NUM_FRAMES:-81}"
GUIDE_SCALE="${GUIDE_SCALE:-6.0}"
LORA_STRENGTH="${DISTILL_LORA_STRENGTH:-0.75}"
RENOISE_MODE="${RENOISE_MODE:-random}"
STAGE2_USE_EMA="${STAGE2_USE_EMA:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RGB_SR_TILE="${RGB_SR_TILE:-0}"
RGB_SR_TILE_PAD="${RGB_SR_TILE_PAD:-10}"
RGB_SR_PRE_PAD="${RGB_SR_PRE_PAD:-0}"
RGB_SR_FP32="${RGB_SR_FP32:-0}"
MRFLOW_DIRECT_SIGMA="${MRFLOW_DIRECT_SIGMA:-0.12}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
FORCE_CASES=()

# The P0 protocol changed Endpoint-RGB-1HR from the distilled t=250 suffix
# to MrFlow's direct sigma=0.12 correction. Do not let --skip-existing mix
# legacy videos with a newly written config: regenerate only this one case.
rgb1_config="${OUT_ROOT}/configs/endpoint_rgb_1hr.json"
rgb1_videos="${OUT_ROOT}/videos/endpoint_rgb_1hr"
if [[ "${MODE}" == "run" && -d "${rgb1_videos}" ]] && \
   find "${rgb1_videos}" -maxdepth 1 -type f -name '*.mp4' -size +1023c -print -quit | grep -q .; then
  if ! "${WAN_PYTHON}" - "${rgb1_config}" "${MRFLOW_DIRECT_SIGMA}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = float(sys.argv[2])
try:
    actual = float(json.loads(path.read_text(encoding="utf-8"))["wan_final_refine_sigma"])
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if abs(actual - expected) < 1e-12 else 1)
PY
  then
    FORCE_CASES=(endpoint_rgb_1hr)
    echo "P0 migration: legacy Endpoint-RGB-1HR detected; only this case will be regenerated."
  fi
fi

[[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
if [[ ! -f "${REALESRGAN_X2_CHECKPOINT}" ]]; then
  if [[ "${AUTO_DOWNLOAD_REALESRGAN}" != "1" ]]; then
    echo "Real-ESRGAN checkpoint is missing and AUTO_DOWNLOAD_REALESRGAN=0: ${REALESRGAN_X2_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ -z "${REALESRGAN_X2_URL}" ]]; then
    echo "Set REALESRGAN_X2_URL or provide REALESRGAN_X2_CKPT." >&2
    exit 1
  fi
  echo "Downloading official RealESRGAN_x2plus.pth to ${REALESRGAN_X2_CHECKPOINT}"
  "${WAN_PYTHON}" - "${REALESRGAN_X2_URL}" "${REALESRGAN_X2_CHECKPOINT}" "${REALESRGAN_X2_BYTES}" <<'PY'
import os
import shutil
import sys
import urllib.request
from pathlib import Path

url, raw_target, raw_expected = sys.argv[1:]
target = Path(raw_target)
expected = int(raw_expected)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
try:
    request = urllib.request.Request(
        url, headers={"User-Agent": "wanUpsampler-distill4-launcher"}
    )
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    actual = temporary.stat().st_size
    if actual != expected:
        raise RuntimeError(f"downloaded {actual} bytes, expected {expected}")
    temporary.replace(target)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
print(f"Downloaded {target} ({expected} bytes)")
PY
fi
actual_realesrgan_bytes="$(wc -c < "${REALESRGAN_X2_CHECKPOINT}" | tr -d ' ')"
if [[ "${actual_realesrgan_bytes}" != "${REALESRGAN_X2_BYTES}" ]]; then
  echo "Unexpected RealESRGAN_x2plus.pth size: ${actual_realesrgan_bytes}; expected ${REALESRGAN_X2_BYTES}" >&2
  exit 1
fi
for directory in "${LIGHTX2V_REPO}" "${MODEL_ROOT}"; do
  [[ -d "${directory}" ]] || { echo "Directory not found: ${directory}" >&2; exit 1; }
done
for file in \
  "${DIT_CKPT}" \
  "${STAGE2_CHECKPOINT}" \
  "${STAGE2_TRAIN_CONFIG}" \
  "${LORA_CHECKPOINT}" \
  "${REALESRGAN_X2_CHECKPOINT}" \
  "${PROMPTS_FILE}"; do
  [[ -f "${file}" ]] || { echo "File not found: ${file}" >&2; exit 1; }
done

prompt_count="$(grep -Ev '^[[:space:]]*(#|$)' "${PROMPTS_FILE}" | wc -l | tr -d ' ')"
if (( prompt_count < PROMPT_OFFSET + LIMIT )); then
  echo "Need $((PROMPT_OFFSET + LIMIT)) prompts, found ${prompt_count}: ${PROMPTS_FILE}" >&2
  exit 1
fi

python_roots=("${LIGHTX2V_REPO}" "${DIFFSYNTH_REPO}" "${PROJECT_ROOT}")
if [[ -d "${REALESRGAN_REPO}" ]]; then
  python_roots+=("${REALESRGAN_REPO}")
fi
joined_pythonpath="$(IFS=:; echo "${python_roots[*]}")"
export PYTHONPATH="${joined_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"
export DTYPE="${DTYPE:-BF16}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${WAN_PYTHON}" - <<'PY'
import torch
from changing_resolution_distill.realesrgan_compat import install_functional_tensor_shim

install_functional_tensor_shim()
import basicsr  # noqa: F401
import realesrgan  # noqa: F401

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"Expected four visible CUDA devices, found {count}")
print(f"Preflight passed: visible_cuda_devices={count}; Real-ESRGAN/BasicSR importable")
PY

mkdir -p "${OUT_ROOT}/logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${OUT_ROOT}/logs/${MODE}_${timestamp}.log"
launch_path="${OUT_ROOT}/logs/${MODE}_${timestamp}_resolved.env"
{
  printf 'PROJECT_ROOT=%q\n' "${PROJECT_ROOT}"
  printf 'WAN_PYTHON=%q\n' "${WAN_PYTHON}"
  printf 'LIGHTX2V_REPO=%q\n' "${LIGHTX2V_REPO}"
  printf 'MODEL_ROOT=%q\n' "${MODEL_ROOT}"
  printf 'DIT_CKPT=%q\n' "${DIT_CKPT}"
  printf 'STAGE2_CHECKPOINT=%q\n' "${STAGE2_CHECKPOINT}"
  printf 'STAGE2_TRAIN_CONFIG=%q\n' "${STAGE2_TRAIN_CONFIG}"
  printf 'LORA_CHECKPOINT=%q\n' "${LORA_CHECKPOINT}"
  printf 'REALESRGAN_X2_CHECKPOINT=%q\n' "${REALESRGAN_X2_CHECKPOINT}"
  printf 'REALESRGAN_X2_URL=%q\n' "${REALESRGAN_X2_URL}"
  printf 'PROMPTS_FILE=%q\n' "${PROMPTS_FILE}"
  printf 'OUT_ROOT=%q\n' "${OUT_ROOT}"
  printf 'GPU_IDS=%q\n' "${GPU_IDS}"
  printf 'SEED=%q\n' "${SEED}"
  printf 'LIMIT=%q\n' "${LIMIT}"
  printf 'LORA_STRENGTH=%q\n' "${LORA_STRENGTH}"
  printf 'MRFLOW_DIRECT_SIGMA=%q\n' "${MRFLOW_DIRECT_SIGMA}"
} > "${launch_path}"

command=(
  "${WAN_PYTHON}"
  "${PROJECT_ROOT}/paper/aaai27/experiments/run_distill4_quality_efficiency.py"
  "${MODE}"
  --out-root "${OUT_ROOT}"
  --prompts "${PROMPTS_FILE}"
  --model-root "${MODEL_ROOT}"
  --dit-ckpt "${DIT_CKPT}"
  --stage2-checkpoint "${STAGE2_CHECKPOINT}"
  --stage2-train-config "${STAGE2_TRAIN_CONFIG}"
  --lora-checkpoint "${LORA_CHECKPOINT}"
  --lora-strength "${LORA_STRENGTH}"
  --realesrgan-x2-checkpoint "${REALESRGAN_X2_CHECKPOINT}"
  --rgb-sr-backend realesrgan
  --rgb-sr-tile "${RGB_SR_TILE}"
  --rgb-sr-tile-pad "${RGB_SR_TILE_PAD}"
  --rgb-sr-pre-pad "${RGB_SR_PRE_PAD}"
  --mrflow-direct-sigma "${MRFLOW_DIRECT_SIGMA}"
  --case-groups native handoff endpoint
  --endpoint-refinement-steps 0 1 2 4
  --endpoint-resizers stage2 interp rgb
  --gpus "${GPUS[@]}"
  --seed "${SEED}"
  --limit "${LIMIT}"
  --prompt-offset "${PROMPT_OFFSET}"
  --num-frames "${NUM_FRAMES}"
  --guide-scale "${GUIDE_SCALE}"
  --renoise-mode "${RENOISE_MODE}"
  --negative-prompt "${NEGATIVE_PROMPT}"
  --python "${WAN_PYTHON}"
)
if [[ "${STAGE2_USE_EMA}" == "1" ]]; then
  command+=(--stage2-use-ema)
else
  command+=(--no-stage2-use-ema)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  command+=(--skip-existing)
else
  command+=(--no-skip-existing)
fi
if [[ "${RGB_SR_FP32}" == "1" ]]; then
  command+=(--rgb-sr-fp32)
fi
if (( ${#FORCE_CASES[@]} > 0 )); then
  command+=(--force-cases "${FORCE_CASES[@]}")
fi

echo "Distill4 final suite: 18 cases x ${LIMIT} prompts; GPUs=${GPU_IDS}"
echo "Output: ${OUT_ROOT}"
echo "Log:    ${log_path}"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}" 2>&1 | tee "${log_path}"

if [[ "${MODE}" == "run" ]]; then
  expected=$((18 * LIMIT))
  actual="$(find "${OUT_ROOT}/videos" -mindepth 2 -maxdepth 2 -type f -name '*.mp4' -size +1023c | wc -l | tr -d ' ')"
  if (( actual < expected )); then
    echo "Run finished but only ${actual}/${expected} valid videos were found." >&2
    exit 1
  fi
  echo "Completed: ${actual} valid videos; schedule=${OUT_ROOT}/generation_schedule.json"
fi
