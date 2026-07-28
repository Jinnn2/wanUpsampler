#!/usr/bin/env bash
# Export the still-missing dataset, split, source-revision, and machine records.

set -euo pipefail

usage() {
  echo "Usage: bash tools/export_missing_repro_metadata.sh --project-root PATH --output PATH"
}

PROJECT_ROOT=""
OUTPUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --output) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "${PROJECT_ROOT}" && -n "${OUTPUT_ROOT}" ]] || { usage >&2; exit 2; }

PROJECT_ROOT="$(realpath "${PROJECT_ROOT}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${WAN_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 1
fi
OUTPUT_PARENT="$(realpath -m "$(dirname "${OUTPUT_ROOT}")")"
OUTPUT_ROOT="${OUTPUT_PARENT}/$(basename "${OUTPUT_ROOT}")"
[[ "${OUTPUT_ROOT}" != "/" && "${OUTPUT_ROOT}" != "${PROJECT_ROOT}" ]] || {
  echo "Unsafe output path: ${OUTPUT_ROOT}" >&2
  exit 1
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "Output already exists: ${OUTPUT_ROOT}" >&2
  exit 1
}
mkdir -p "${OUTPUT_ROOT}/environment" "${OUTPUT_ROOT}/git"

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_lmdb_metadata.py" \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT_ROOT}/dataset_lmdb_manifest.json" || true

{
  date --iso-8601=seconds
  uname -a
  [[ -f /etc/os-release ]] && cat /etc/os-release
} > "${OUTPUT_ROOT}/environment/os.txt"

{
  command -v lscpu >/dev/null 2>&1 && lscpu
  command -v free >/dev/null 2>&1 && free -b
} > "${OUTPUT_ROOT}/environment/cpu_memory.txt"

{
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi
  command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
      --format=csv,noheader
} > "${OUTPUT_ROOT}/environment/nvidia_smi.txt"

{
  command -v nvcc >/dev/null 2>&1 && nvcc --version
  command -v gcc >/dev/null 2>&1 && gcc --version
  command -v ffmpeg >/dev/null 2>&1 && ffmpeg -version
} > "${OUTPUT_ROOT}/environment/toolchain.txt"

"${PYTHON_BIN}" -VV > "${OUTPUT_ROOT}/environment/python.txt" 2>&1
"${PYTHON_BIN}" -m pip freeze --all > "${OUTPUT_ROOT}/environment/pip_freeze.txt"
if command -v conda >/dev/null 2>&1; then
  conda env export > "${OUTPUT_ROOT}/environment/conda_environment.yml"
  conda list --explicit > "${OUTPUT_ROOT}/environment/conda_explicit.txt"
  conda env list --json > "${OUTPUT_ROOT}/environment/conda_env_list.json"
fi

"${PYTHON_BIN}" - > "${OUTPUT_ROOT}/environment/python_runtime.json" <<'PY'
import importlib
import json
import platform
import sys

names = [
    "torch", "torchvision", "transformers", "diffusers", "accelerate",
    "safetensors", "numpy", "scipy", "cv2", "lpips", "lmdb",
    "lightx2v", "lightx2v_platform",
]
out = {"python": sys.version, "platform": platform.platform(), "modules": {}}
for name in names:
    try:
        module = importlib.import_module(name)
        out["modules"][name] = {"version": getattr(module, "__version__", None)}
    except Exception as exc:
        out["modules"][name] = {"error": repr(exc)}
try:
    import torch
    out["torch_runtime"] = {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except Exception:
    pass
print(json.dumps(out, indent=2, sort_keys=True))
PY

if [[ -n "${VBENCH_PYTHON:-}" && -x "${VBENCH_PYTHON}" ]]; then
  mkdir -p "${OUTPUT_ROOT}/environment/vbench"
  "${VBENCH_PYTHON}" -VV \
    > "${OUTPUT_ROOT}/environment/vbench/python.txt" 2>&1
  "${VBENCH_PYTHON}" -m pip freeze --all \
    > "${OUTPUT_ROOT}/environment/vbench/pip_freeze.txt"
  PYTHONPATH="${VBENCH_REPO:-}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${VBENCH_PYTHON}" - > "${OUTPUT_ROOT}/environment/vbench/runtime.json" <<'PY'
import importlib
import json
import platform
import sys

out = {"python": sys.version, "platform": platform.platform(), "modules": {}}
for name in ["torch", "torchvision", "numpy", "scipy", "cv2", "vbench"]:
    try:
        module = importlib.import_module(name)
        out["modules"][name] = {"version": getattr(module, "__version__", None)}
    except Exception as exc:
        out["modules"][name] = {"error": repr(exc)}
print(json.dumps(out, indent=2, sort_keys=True))
PY
fi

capture_git() {
  local name="$1"
  local repo="$2"
  if [[ -z "${repo}" || ! -d "${repo}/.git" ]]; then
    echo "not_available" > "${OUTPUT_ROOT}/git/${name}.status"
    return
  fi
  git -C "${repo}" rev-parse HEAD > "${OUTPUT_ROOT}/git/${name}.commit"
  git -C "${repo}" status --porcelain=v2 > "${OUTPUT_ROOT}/git/${name}.status"
  git -C "${repo}" diff --binary HEAD > "${OUTPUT_ROOT}/git/${name}.patch"
  git -C "${repo}" submodule status --recursive \
    > "${OUTPUT_ROOT}/git/${name}.submodules" 2>&1 || true
}
capture_git "wanupsampler" "${PROJECT_ROOT}"
capture_git "lightx2v" "${LIGHTX2V_REPO:-}"
capture_git "diffsynth" "${DIFFSYNTH_REPO:-}"
capture_git "vbench" "${VBENCH_REPO:-}"
capture_git "realesrgan" "${REALESRGAN_REPO:-}"

(
  cd "${OUTPUT_ROOT}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
echo "metadata export ready: ${OUTPUT_ROOT}"
