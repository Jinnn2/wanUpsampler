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
DIFFSYNTH_REF="${DIFFSYNTH_REF:-main}"
DIFFSYNTH_GIT_URL="${DIFFSYNTH_GIT_URL:-https://github.com/modelscope/DiffSynth-Studio.git}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-${JIN_ROOT}/LightX2V}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_GIT_UPDATE="${SKIP_GIT_UPDATE:-0}"
INSTALL_TRAIN_EXTRAS="${INSTALL_TRAIN_EXTRAS:-1}"

MODE="${1:-install}"

clone_or_update_diffsynth() {
  if [[ ! -d "${DIFFSYNTH_REPO}/.git" ]]; then
    mkdir -p "$(dirname "${DIFFSYNTH_REPO}")"
    git clone "${DIFFSYNTH_GIT_URL}" "${DIFFSYNTH_REPO}"
  elif [[ "${SKIP_GIT_UPDATE}" != "1" ]]; then
    git -C "${DIFFSYNTH_REPO}" fetch --tags origin
  fi

  git -C "${DIFFSYNTH_REPO}" checkout "${DIFFSYNTH_REF}"
}

install_python_packages() {
  "${PYTHON_BIN}" -m pip install -e "${DIFFSYNTH_REPO}"

  if [[ "${INSTALL_TRAIN_EXTRAS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install accelerate modelscope deepspeed safetensors peft
  fi
}

show_summary() {
  cat <<EOF
Last-step-skip LoRA environment
  project          : ${PROJECT_ROOT}
  DiffSynth repo   : ${DIFFSYNTH_REPO}
  DiffSynth ref    : ${DIFFSYNTH_REF}
  LightX2V repo    : ${LIGHTX2V_REPO}
  python           : ${PYTHON_BIN}
  train extras     : ${INSTALL_TRAIN_EXTRAS}

Recommended shell exports for manual runs:
  export PYTHONPATH="${DIFFSYNTH_REPO}:${LIGHTX2V_REPO}:${PROJECT_ROOT}:\${PYTHONPATH:-}"
EOF
}

case "${MODE}" in
  install)
    clone_or_update_diffsynth
    install_python_packages
    show_summary
    bash "${SCRIPT_DIR}/check_last_step_skip_lora_env.sh"
    ;;
  check)
    show_summary
    bash "${SCRIPT_DIR}/check_last_step_skip_lora_env.sh"
    ;;
  *)
    echo "Usage: bash changing_resolution_distill/scripts/train/setup_last_step_skip_lora_env.sh [install|check]" >&2
    exit 2
    ;;
esac
