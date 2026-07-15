#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
LMDB_DIR="${CR_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CHECKPOINT="${CR_STAGE2_OPERATOR_COMPARE_CKPT:-${CR_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p_stage2_lmdb}/latest.pt}"
TRAIN_CONFIG="${CR_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml}"
OUT_DIR="${CR_STAGE2_OPERATOR_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_operator_compare_stage2}"

CALLER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
GPU_IDS="${GPU_IDS:-${CALLER_CUDA_VISIBLE_DEVICES}}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-32}"
SPLIT="${SPLIT:-val}"
PRECISION="${PRECISION:-bf16}"
USE_EMA="${USE_EMA:-0}"
STAGE2_RESIDUAL_SKIP="${STAGE2_RESIDUAL_SKIP:-checkpoint}"
FPS="${FPS:-16}"
METRICS="${METRICS:-psnr ssim lpips}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-4}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/changing_resolution_stage2_operator_compare}"

export LIGHTX2V_REPO
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${GPU_IDS}" ]]; then
  visible_gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if (( visible_gpu_count < 1 )); then
    echo "CUDA is unavailable in the operator-compare environment." >&2
    exit 1
  fi
  GPU_IDS="$(seq -s, 0 $((visible_gpu_count - 1)))"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${LMDB_DIR}" ]]; then
  echo "LMDB dir not found: ${LMDB_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${#GPUS[@]}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

base_count=$((TOTAL_SAMPLES / NUM_GPUS))
remainder=$((TOTAL_SAMPLES % NUM_GPUS))
offset=0
pids=()
part_names=()

for rank in "${!GPUS[@]}"; do
  count="${base_count}"
  if (( rank < remainder )); then
    count=$((count + 1))
  fi
  if (( count == 0 )); then
    continue
  fi

  gpu="${GPUS[$rank]}"
  part_name="$(printf "part_%02d" "${rank}")"
  part_out="${OUT_DIR}/${part_name}"
  log_path="${LOG_DIR}/${part_name}.log"
  ema_args=()
  if [[ "${USE_EMA}" == "1" ]]; then
    ema_args=(--use_ema)
  fi

  echo "Launch ${part_name}: gpu=${gpu}, split=${SPLIT}, offset=${offset}, count=${count}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python changing_resolution/scripts/eval/eval_clean_resizer_operator_compare.py \
      --data_dir "${LMDB_DIR}" \
      --data_format lmdb \
      --checkpoint "${CHECKPOINT}" \
      --train_config "${TRAIN_CONFIG}" \
      --model_root "${MODEL_ROOT}" \
      --vae_path "${VAE_PATH}" \
      --wan_repo "${LIGHTX2V_REPO}" \
      --out_dir "${part_out}" \
      --split "${SPLIT}" \
      --offset "${offset}" \
      --limit "${count}" \
      --precision "${PRECISION}" \
      --metrics ${METRICS} \
      --metric_batch_size "${METRIC_BATCH_SIZE}" \
      --fps "${FPS}" \
      --model_class stage2 \
      --stage2_residual_skip "${STAGE2_RESIDUAL_SKIP}" \
      "${ema_args[@]}"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  part_names+=("${part_name}")
  offset=$((offset + count))
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "Stage 2 operator compare failed. Check logs under: ${LOG_DIR}" >&2
  for log_path in "${LOG_DIR}"/part_*.log; do
    [[ -f "${log_path}" ]] || continue
    echo "===== ${log_path} (last 80 lines) =====" >&2
    tail -n 80 "${log_path}" >&2
  done
  exit 1
fi

merged_metrics="${OUT_DIR}/metrics_${SPLIT}.jsonl"
: > "${merged_metrics}"
for part_name in "${part_names[@]}"; do
  cat "${OUT_DIR}/${part_name}"/metrics_"${SPLIT}"_*.jsonl >> "${merged_metrics}"
done
summary_path="${OUT_DIR}/summary_${SPLIT}.json"
python - "${merged_metrics}" "${summary_path}" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
numeric = {}
for row in rows:
    for key, value in row.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            numeric.setdefault(key, []).append(float(value))
summary = {
    "num_samples": len(rows),
    "mean": {key: sum(values) / len(values) for key, values in sorted(numeric.items()) if values},
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary["mean"], ensure_ascii=False, indent=2))
PY
echo "Stage 2 operator compare ready: ${OUT_DIR}"
echo "Merged metrics: ${merged_metrics}"
echo "Summary: ${summary_path}"
