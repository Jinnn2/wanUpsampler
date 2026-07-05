#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

OUT_ROOT="${OUT_ROOT:-/mnt/afs_2/houze/wanUpsampler/outputs/changing_resolution_distill_last_step_skip_lora_clean_pred_compare_480p}"

python "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/eval_last_step_skip_lora_video_closeness.py" \
  --out_root "${OUT_ROOT}" \
  "$@"
