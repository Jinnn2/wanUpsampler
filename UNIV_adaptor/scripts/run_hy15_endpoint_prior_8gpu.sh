#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-plan}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-/mnt/afs_2/houze/envs/hy15_endpoint}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/opt/conda/bin/python}"
HY_PYTHON="${HY_PYTHON:-${ENV_ROOT}/bin/python}"
ASSETS_ROOT="${ASSETS_ROOT:-/mnt/afs_2/houze/models/hy15_endpoint_v1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/hy15_endpoint_prior_v1}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DRIVER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/run_hy15_endpoint_prior.py"
SCORER="${PROJECT_ROOT}/UNIV_adaptor/scripts/data/score_hy15_endpoint_prior.py"
COMMON=(--out "${OUT_ROOT}" --assets "${ASSETS_ROOT}")
[[ -z "${PROTOCOL:-}" ]] || COMMON+=(--protocol "${PROTOCOL}")
[[ -z "${PROMPTS_JSONL:-}" ]] || COMMON+=(--prompts "${PROMPTS_JSONL}")

if [[ "${MODE}" == setup ]]; then
  [[ -x "${HY_PYTHON}" ]] || "${BOOTSTRAP_PYTHON}" -m venv "${ENV_ROOT}"
  "${HY_PYTHON}" -m pip install --upgrade pip
  "${HY_PYTHON}" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
  "${HY_PYTHON}" -m pip install diffusers==0.36.0 transformers==4.57.3 accelerate==1.12.0 \
    huggingface-hub==0.36.0 safetensors==0.7.0 sentencepiece==0.2.1 \
    numpy==1.26.4 imageio==2.37.2 imageio-ffmpeg==0.6.0 \
    realesrgan==0.3.0 basicsr==1.4.2 opencv-python==4.11.0.86 loguru==0.7.3
  "${HY_PYTHON}" -m pip check
  exit 0
fi
if [[ "${MODE}" == plan ]]; then
  "${BOOTSTRAP_PYTHON}" "${DRIVER}" plan "${COMMON[@]}"
  exit 0
fi
if [[ "${MODE}" == download ]]; then
  "${HY_PYTHON}" "${DRIVER}" download "${COMMON[@]}"
  exit 0
fi
if [[ "${MODE}" == partial-check ]]; then
  # Read-only CPU audit may run while the generation launcher owns its lock.
  "${HY_PYTHON}" "${SCORER}" partial-check --out "${OUT_ROOT}" --vbench-root "${VBENCH_ROOT}"
  exit 0
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ ${#GPUS[@]} == 8 ]] || { echo "Exactly 8 GPU ids required" >&2; exit 2; }
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ && -z "${SEEN[$gpu]:-}" ]] || { echo "Invalid/duplicate GPU id" >&2; exit 2; }
  SEEN[$gpu]=1
done
mkdir -p "${OUT_ROOT}/logs"
# Kernel-held lock is automatically released after a crash; do not unlink it.
exec 9>"${OUT_ROOT}/.launcher.lock"
flock -n 9 || { echo "Another generation/scoring launcher owns this root" >&2; exit 1; }

case "${MODE}" in
  check)
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${HY_PYTHON}" "${DRIVER}" check "${COMMON[@]}" ;;
  smoke|generate)
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${HY_PYTHON}" "${DRIVER}" freeze "${COMMON[@]}"
    if [[ "${MODE}" == smoke ]]; then
      CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${HY_PYTHON}" "${DRIVER}" worker "${COMMON[@]}" --rank 0 --world 8 --limit 8 \
        2>&1 | tee "${OUT_ROOT}/logs/smoke.log"
    else
      PIDS=()
      trap 'for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
      for rank in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "${HY_PYTHON}" "${DRIVER}" worker "${COMMON[@]}" --rank "${rank}" --world 8 \
          >"${OUT_ROOT}/logs/gpu_${rank}.log" 2>&1 &
        PIDS+=("$!")
      done
      FAILED=0
      for pid in "${PIDS[@]}"; do wait "${pid}" || FAILED=1; done
      [[ "${FAILED}" == 0 ]] || { echo "Worker failed; inspect logs; generation can resume" >&2; exit 1; }
      "${HY_PYTHON}" "${DRIVER}" finalize "${COMMON[@]}"
    fi ;;
  finalize) "${HY_PYTHON}" "${DRIVER}" finalize "${COMMON[@]}" ;;
  score-check|score|report|partial-score|partial-report)
    SCORE_MODE="${MODE}"
    [[ "${MODE}" != score-check ]] || SCORE_MODE=check
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${HY_PYTHON}" "${SCORER}" "${SCORE_MODE}" --out "${OUT_ROOT}" \
      --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" \
      --expected-vbench-commit "${EXPECTED_VBENCH_COMMIT:-}" --ngpus 8 ;;
  *) echo "Usage: $0 [setup|download|plan|check|smoke|generate|finalize|score-check|score|report|partial-check|partial-score|partial-report]" >&2; exit 2 ;;
esac
