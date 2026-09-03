#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  check|prepare|generate|visualize|vbench|summarize|all) ;;
  *)
    echo "Usage: $0 [check|prepare|generate|visualize|vbench|summarize|all]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
REALESRGAN_REPO="${REALESRGAN_REPO:-/mnt/afs_2/houze/Real-ESRGAN}"
REALESRGAN_X2_CKPT="${REALESRGAN_X2_CKPT:-${REALESRGAN_REPO}/weights/RealESRGAN_x2plus.pth}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"

PROFILE="${PROFILE:-core}"
LIMIT="${LIMIT:-10}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SEED="${SEED:-9700}"
TIMING_WARMUP="${TIMING_WARMUP:-1}"
GPU_ID="${GPU_ID:-0}"
VBENCH_GPU_IDS="${VBENCH_GPU_IDS:-0}"
VBENCH_NGPUS="${VBENCH_NGPUS:-1}"
VBENCH_COMMIT="${VBENCH_COMMIT:-}"
RESUME="${RESUME:-1}"
FORCE_VBENCH="${FORCE_VBENCH:-0}"
SKIP_VBENCH_WARMUP="${SKIP_VBENCH_WARMUP:-0}"
ENABLE_TRANSITION_DIAGNOSTICS="${ENABLE_TRANSITION_DIAGNOSTICS:-0}"
CASE_NAME="${CASE_NAME:-}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/univ_validation_${PROFILE}_${LIMIT}p}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-camera shake, overexposed, static image, blurry details, subtitles, text, watermark, low quality, jpeg artifacts, distorted hands, distorted face, malformed body, duplicate limbs}"
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/validation/run_univ_validation.py"
FFMPEG="${FFMPEG:-ffmpeg}"

resolve_vbench_python() {
  if [[ -n "${VBENCH_PYTHON}" ]]; then
    [[ -x "${VBENCH_PYTHON}" ]] || {
      echo "VBENCH_PYTHON is not executable: ${VBENCH_PYTHON}" >&2
      exit 1
    }
    return
  fi

  local candidate
  for candidate in \
    /opt/conda/envs/vbench/bin/python \
    /opt/conda/bin/python \
    "$(command -v python 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if (
      cd "${VBENCH_ROOT}"
      "${candidate}" -c "import torch, vbench" >/dev/null 2>&1
    ); then
      VBENCH_PYTHON="${candidate}"
      echo "Auto-selected VBENCH_PYTHON=${VBENCH_PYTHON}"
      return
    fi
  done

  echo "No Python environment can import torch and vbench from ${VBENCH_ROOT}." >&2
  echo "Set VBENCH_PYTHON explicitly after installing VBench dependencies." >&2
  exit 1
}

[[ -x "${WAN_PYTHON}" ]] || { echo "Python is not executable: ${WAN_PYTHON}" >&2; exit 1; }
[[ -f "${DRIVER}" ]] || { echo "Required file not found: ${DRIVER}" >&2; exit 1; }
if [[ "${MODE}" == "check" || "${MODE}" == "prepare" || "${MODE}" == "generate" || "${MODE}" == "all" ]]; then
  [[ -f "${PROMPTS_FILE}" ]] || { echo "Required file not found: ${PROMPTS_FILE}" >&2; exit 1; }
fi
if [[ "${MODE}" == "check" || "${MODE}" == "generate" || "${MODE}" == "all" ]]; then
  for directory in "${LIGHTX2V_REPO}" "${REALESRGAN_REPO}" "${MODEL_ROOT}"; do
    [[ -d "${directory}" ]] || { echo "Required directory not found: ${directory}" >&2; exit 1; }
  done
  [[ -f "${REALESRGAN_X2_CKPT}" ]] || { echo "Required file not found: ${REALESRGAN_X2_CKPT}" >&2; exit 1; }
fi
if [[ "${MODE}" == "check" || "${MODE}" == "vbench" || "${MODE}" == "all" ]]; then
  [[ -d "${VBENCH_ROOT}" ]] || { echo "Required directory not found: ${VBENCH_ROOT}" >&2; exit 1; }
  [[ -f "${VBENCH_ROOT}/evaluate.py" ]] || { echo "Required file not found: ${VBENCH_ROOT}/evaluate.py" >&2; exit 1; }
  resolve_vbench_python
fi
VBENCH_PYTHON="${VBENCH_PYTHON:-${WAN_PYTHON}}"
if [[ "${MODE}" == "check" || "${MODE}" == "visualize" || "${MODE}" == "all" ]]; then
  command -v "${FFMPEG}" >/dev/null 2>&1 || { echo "ffmpeg not found: ${FFMPEG}" >&2; exit 1; }
fi

export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${REALESRGAN_REPO}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DTYPE="${DTYPE:-BF16}"

common_args=(
  --profile "${PROFILE}"
  --out-root "${OUT_ROOT}"
  --prompts "${PROMPTS_FILE}"
  --limit "${LIMIT}"
  --prompt-offset "${PROMPT_OFFSET}"
  --seed "${SEED}"
  --timing-warmup "${TIMING_WARMUP}"
  --gpu "${GPU_ID}"
  --model-root "${MODEL_ROOT}"
  --lightx2v-repo "${LIGHTX2V_REPO}"
  --realesrgan-repo "${REALESRGAN_REPO}"
  --realesrgan-checkpoint "${REALESRGAN_X2_CKPT}"
  --wan-python "${WAN_PYTHON}"
  --vbench-root "${VBENCH_ROOT}"
  --vbench-python "${VBENCH_PYTHON}"
  --vbench-ngpus "${VBENCH_NGPUS}"
  --ffmpeg "${FFMPEG}"
  --negative-prompt "${NEGATIVE_PROMPT}"
)
if [[ "${RESUME}" == "1" ]]; then
  common_args+=(--resume)
fi
if [[ "${FORCE_VBENCH}" == "1" ]]; then
  common_args+=(--force-vbench)
fi
if [[ "${SKIP_VBENCH_WARMUP}" == "1" ]]; then
  common_args+=(--skip-vbench-warmup)
fi
if [[ "${ENABLE_TRANSITION_DIAGNOSTICS}" == "1" ]]; then
  common_args+=(--transition-diagnostics)
else
  common_args+=(--no-transition-diagnostics)
fi
if [[ -n "${VBENCH_COMMIT}" ]]; then
  common_args+=(--vbench-commit "${VBENCH_COMMIT}")
fi
if [[ -n "${CASE_NAME}" ]]; then
  common_args+=(--case-name "${CASE_NAME}")
fi

run_prepare() {
  "${WAN_PYTHON}" "${DRIVER}" prepare "${common_args[@]}"
}

run_generate() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${WAN_PYTHON}" "${DRIVER}" generate "${common_args[@]}"
}

run_vbench() {
  CUDA_VISIBLE_DEVICES="${VBENCH_GPU_IDS}" \
    "${WAN_PYTHON}" "${DRIVER}" vbench "${common_args[@]}"
}

run_visualize() {
  "${WAN_PYTHON}" "${DRIVER}" visualize "${common_args[@]}"
}

run_summarize() {
  "${WAN_PYTHON}" "${DRIVER}" summarize "${common_args[@]}"
}

if [[ "${MODE}" == "check" ]]; then
  run_prepare
  "${WAN_PYTHON}" - "${MODEL_ROOT}" "${REALESRGAN_X2_CKPT}" <<'PY'
import sys
from pathlib import Path

from changing_resolution_distill.realesrgan_compat import install_functional_tensor_shim
from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root

install_functional_tensor_shim()
import basicsr  # noqa: F401
import realesrgan  # noqa: F401
from lightx2v.common.ops import *  # noqa: F403
import UNIV_adaptor.wan_runner  # noqa: F401
from lightx2v.utils.registry_factory import RUNNER_REGISTER

config = validate_wan21_t2v_model_root(sys.argv[1])
checkpoint = Path(sys.argv[2])
if checkpoint.stat().st_size != 67061725:
    raise SystemExit(f"Unexpected RealESRGAN_x2plus.pth size: {checkpoint.stat().st_size}")
print(f"Wan model contract: dim={config['dim']}, heads={config['num_heads']}")
print(f"Real-ESRGAN checkpoint: {checkpoint} ({checkpoint.stat().st_size} bytes)")
for runner_name in ("wan2.1_univ_native", "wan2.1_univ_pipeline"):
    if runner_name not in RUNNER_REGISTER:
        raise SystemExit(f"Runner registration missing: {runner_name}")
print("LightX2V, UNIV runners, BasicSR, and Real-ESRGAN imports passed")
PY
  (
    cd "${VBENCH_ROOT}"
  "${VBENCH_PYTHON}" -c "import torch, vbench; print('VBench imports passed; CUDA:', torch.cuda.is_available())"
  )
  echo "UNIV validation preflight passed: ${OUT_ROOT}"
  exit 0
fi

case "${MODE}" in
  prepare) run_prepare ;;
  generate) run_generate ;;
  visualize) run_visualize ;;
  vbench) run_vbench ;;
  summarize) run_summarize ;;
  all)
    run_prepare
    run_generate
    run_visualize
    run_vbench
    run_summarize
    ;;
esac

echo "Videos : ${OUT_ROOT}/videos"
echo "Timing : ${OUT_ROOT}/timings"
echo "Compare: ${OUT_ROOT}/comparisons"
echo "VBench : ${OUT_ROOT}/metrics/vbench_scores.json"
echo "Summary: ${OUT_ROOT}/reports/SUMMARY.md"
