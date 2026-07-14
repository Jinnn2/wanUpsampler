#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

export CR_STAGE2_LMDB_DIR="${CR_DISTILL_360_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_clean_368x640_720x1248_14b_cfgdistill_5k}"
export CR_DISTILL_LORA_LMDB_DIR="${CR_DISTILL_360_LORA_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3}"
export CR_DISTILL_STAGE3_X0PRED_CONFIG="${CR_DISTILL_360_LORA_TRAJECTORY_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_368x640.json}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 4 )); then
  echo "This builder requires exactly four GPU IDs; got GPU_IDS=${GPU_IDS}." >&2
  exit 2
fi

exec bash "${PROJECT_ROOT}/changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb_multigpu.sh"
