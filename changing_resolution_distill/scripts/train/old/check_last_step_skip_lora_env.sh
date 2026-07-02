#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

JIN_ROOT="${JIN_ROOT:-/mnt/afs_2/houze}"
DIFFSYNTH_REPO="${DIFFSYNTH_REPO:-${JIN_ROOT}/DiffSynth-Studio}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-${JIN_ROOT}/LightX2V}"
CR_DISTILL_MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-${JIN_ROOT}/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CR_DISTILL_DIT_CKPT="${CR_DISTILL_DIT_CKPT:-${CR_DISTILL_MODEL_ROOT}/distill_model.pt}"
CR_DISTILL_LORA_CONFIG="${CR_DISTILL_LORA_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml}"
CR_DISTILL_LORA_LMDB_DIR="${CR_DISTILL_LORA_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3}"

PYTHON_BIN="${PYTHON_BIN:-python}"

missing=0

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    missing=1
  else
    echo "ok ${label}: ${path}"
  fi
}

require_path "project" "${PROJECT_ROOT}"
require_path "DiffSynth-Studio repo" "${DIFFSYNTH_REPO}"
require_path "LightX2V repo" "${LIGHTX2V_REPO}"
require_path "distill model root" "${CR_DISTILL_MODEL_ROOT}"
require_path "distill DiT checkpoint" "${CR_DISTILL_DIT_CKPT}"
require_path "LoRA config" "${CR_DISTILL_LORA_CONFIG}"

TRAIN_PY="${DIFFSYNTH_REPO}/examples/wanvideo/model_training/train.py"
ACCELERATE_14B="${DIFFSYNTH_REPO}/examples/wanvideo/model_training/full/accelerate_config_14B.yaml"
require_path "DiffSynth Wan train.py" "${TRAIN_PY}"
require_path "DiffSynth 14B accelerate config" "${ACCELERATE_14B}"

if [[ -d "${CR_DISTILL_LORA_LMDB_DIR}" ]]; then
  echo "ok LoRA LMDB exists: ${CR_DISTILL_LORA_LMDB_DIR}"
else
  echo "pending LoRA LMDB: ${CR_DISTILL_LORA_LMDB_DIR}"
  echo "  Build this later with build_last_step_skip_lora_lmdb.py from the implementation plan."
fi

if [[ "${missing}" -ne 0 ]]; then
  echo "LoRA environment path preflight failed." >&2
  exit 1
fi

PYTHONPATH="${DIFFSYNTH_REPO}:${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${TRAIN_PY}" <<'PY'
import importlib.util
import pathlib
import sys

train_py = pathlib.Path(sys.argv[1])
modules = ["torch", "accelerate", "diffsynth", "modelscope", "yaml", "safetensors"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing python packages: "
        + ", ".join(missing)
        + "\nRun setup_last_step_skip_lora_env.sh install first."
    )

text = train_py.read_text(encoding="utf-8")
required_args = [
    "lora_base_model",
    "lora_target_modules",
    "lora_rank",
    "lora_checkpoint",
    "preset_lora_path",
    "use_gradient_checkpointing_offload",
]
missing_args = [arg for arg in required_args if arg not in text]
if missing_args:
    raise SystemExit("DiffSynth train.py is missing expected LoRA args: " + ", ".join(missing_args))

import torch
print(f"ok python packages: {', '.join(modules)}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
print("DiffSynth Wan LoRA entrypoint exposes expected LoRA/offload arguments.")
PY

echo "Last-step-skip LoRA environment preflight passed."
