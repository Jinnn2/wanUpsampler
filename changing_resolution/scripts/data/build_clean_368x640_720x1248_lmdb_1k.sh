#!/usr/bin/env bash
set -euo pipefail

# Reuse the existing 720x1248 teacher videos and encode a dedicated
# 368x640 -> 720x1248 clean-latent LMDB for the near-2x Stage2 model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

export PROJECT_ROOT
export LR_H="${LR_H:-368}"
export LR_W="${LR_W:-640}"
export HR_H="${HR_H:-720}"
export HR_W="${HR_W:-1248}"
export CR_LMDB_DIR="${CR_LMDB_368X640_720X1248_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_368x640_720x1248_1k}"

if [[ "${LR_H}x${LR_W}" != "368x640" || "${HR_H}x${HR_W}" != "720x1248" ]]; then
  echo "This entrypoint is locked to 368x640 -> 720x1248." >&2
  exit 2
fi

exec bash "${PROJECT_ROOT}/changing_resolution/scripts/data/build_clean_480p720p_lmdb_1k.sh" "${1:-all}"
