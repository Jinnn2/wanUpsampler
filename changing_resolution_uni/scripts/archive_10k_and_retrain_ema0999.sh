#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

OLD_OUT_DIR="${OLD_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_uni_clean_v1_1k}"
STEP_CHECKPOINT="${STEP_CHECKPOINT:-${OLD_OUT_DIR}/step_0010000.pt}"
LAST_CHECKPOINT="${LAST_CHECKPOINT:-${OLD_OUT_DIR}/last.pt}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${PROJECT_ROOT}/outputs/checkpoint_archives/u_itu_clean_v1_1k_step10000}"

CONFIG="${CONFIG:-${PROJECT_ROOT}/changing_resolution_uni/configs/train_universal_clean_ema0999.yaml}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/lmdb_clean_v1_1k}"
NEW_OUT_DIR="${NEW_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_uni_clean_v1_1k_fresh_ema0999}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-10000}"
USE_TMUX="${USE_TMUX:-1}"
SESSION_NAME="${SESSION_NAME:-u_itu_clean_v1_1k_ema0999}"
RUN_LOG="${RUN_LOG:-${NEW_OUT_DIR}/train.log}"

[[ "${MAX_STEPS}" == "10000" ]] || {
  echo "This controlled retraining script requires MAX_STEPS=10000, got ${MAX_STEPS}" >&2
  exit 2
}

for path in "${STEP_CHECKPOINT}" "${LAST_CHECKPOINT}" "${CONFIG}"; do
  [[ -f "${path}" ]] || { echo "Required file not found: ${path}" >&2; exit 2; }
done
[[ -d "${DATA_DIR}" ]] || { echo "Training data directory not found: ${DATA_DIR}" >&2; exit 2; }

shopt -s nullglob
shards=("${DATA_DIR}"/shard_*)
shopt -u nullglob
if (( ${#shards[@]} == 0 )); then
  echo "No shard_* directories found under ${DATA_DIR}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
NUM_GPUS="${#gpu_array[@]}"
if (( NUM_GPUS < 1 )); then
  echo "GPU_IDS must contain at least one device id" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python - "${CONFIG}" <<'PY'
import math
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text(encoding="utf-8"))
train = config["train"]
expected = {
    "max_steps": 10000,
    "lr_scheduler": "cosine",
    "warmup_steps": 500,
    "val_max_samples": 100,
    "eval_every": 500,
}
for key, value in expected.items():
    if train.get(key) != value:
        raise SystemExit(f"{path}: train.{key}={train.get(key)!r}, expected {value!r}")
for key, value in {
    "lr": 1e-4,
    "min_lr": 1e-6,
    "ema_decay": 0.999,
    "val_ratio": 0.10,
}.items():
    actual = float(train.get(key, float("nan")))
    if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{path}: train.{key}={actual!r}, expected {value!r}")
print(f"validated fresh-training config: {path}")
PY

python - "${STEP_CHECKPOINT}" "${LAST_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

import torch


def load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    payload = load(path)
    step = int(payload.get("step", -1))
    if step != 10000:
        raise SystemExit(f"{path} contains step={step}, expected 10000")
    for key in ("model", "optimizer", "ema", "config"):
        if key not in payload:
            raise SystemExit(f"{path} is missing checkpoint field {key!r}")
    print(f"validated checkpoint: {path} step={step}")
PY

mkdir -p "${ARCHIVE_DIR}"

copy_once() {
  local source="$1"
  local target="$2"
  if [[ -f "${target}" ]]; then
    cmp -s "${source}" "${target}" || {
      echo "Archive target exists with different content: ${target}" >&2
      exit 2
    }
    echo "archive already verified: ${target}"
    return
  fi
  cp --reflink=auto --preserve=timestamps "${source}" "${target}"
  cmp -s "${source}" "${target}" || {
    echo "Archive copy verification failed: ${target}" >&2
    exit 2
  }
  echo "archived: ${target}"
}

copy_once "${STEP_CHECKPOINT}" "${ARCHIVE_DIR}/step_00010000.pt"
copy_once "${LAST_CHECKPOINT}" "${ARCHIVE_DIR}/last.pt"
if [[ -f "${OLD_OUT_DIR}/train_config.yaml" ]]; then
  copy_once "${OLD_OUT_DIR}/train_config.yaml" "${ARCHIVE_DIR}/train_config.yaml"
fi
if [[ -f "${OLD_OUT_DIR}/metrics.jsonl" ]]; then
  copy_once "${OLD_OUT_DIR}/metrics.jsonl" "${ARCHIVE_DIR}/metrics.jsonl"
fi
(
  cd "${ARCHIVE_DIR}"
  sha256sum step_00010000.pt last.pt > SHA256SUMS.txt.tmp
  mv SHA256SUMS.txt.tmp SHA256SUMS.txt
  sha256sum --check SHA256SUMS.txt
)

if [[ -d "${NEW_OUT_DIR}" ]] && [[ -n "$(find "${NEW_OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Fresh-training output is not empty; refusing to overwrite: ${NEW_OUT_DIR}" >&2
  exit 2
fi
mkdir -p "${NEW_OUT_DIR}" "$(dirname "${RUN_LOG}")"

echo "Archived checkpoints: ${ARCHIVE_DIR}"
echo "Fresh training config: ${CONFIG}"
echo "Fresh training data:   ${DATA_DIR} (${#shards[@]} shards)"
echo "Fresh training output: ${NEW_OUT_DIR}"
echo "GPUs:                  ${GPU_IDS} (${NUM_GPUS})"
echo "Max steps:             ${MAX_STEPS}"
echo "EMA decay:             0.999"

if [[ "${USE_TMUX}" == "1" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 2; }
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
  fi
  printf -v launch_command \
    'cd %q && CUDA_VISIBLE_DEVICES=%q NUM_GPUS=%q CONFIG=%q DATA_DIR=%q OUT_DIR=%q MAX_STEPS=%q RESUME= bash changing_resolution_uni/scripts/run_train.sh 2>&1 | tee %q' \
    "${PROJECT_ROOT}" "${GPU_IDS}" "${NUM_GPUS}" "${CONFIG}" "${DATA_DIR}" \
    "${NEW_OUT_DIR}" "${MAX_STEPS}" "${RUN_LOG}"
  tmux new-session -d -s "${SESSION_NAME}" "${launch_command}"
  echo "Started tmux session: ${SESSION_NAME}"
  echo "Attach: tmux attach -t ${SESSION_NAME}"
  echo "Log: ${RUN_LOG}"
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  NUM_GPUS="${NUM_GPUS}" \
  CONFIG="${CONFIG}" \
  DATA_DIR="${DATA_DIR}" \
  OUT_DIR="${NEW_OUT_DIR}" \
  MAX_STEPS="${MAX_STEPS}" \
  RESUME="" \
  bash "${PROJECT_ROOT}/changing_resolution_uni/scripts/run_train.sh" 2>&1 | tee "${RUN_LOG}"
fi
