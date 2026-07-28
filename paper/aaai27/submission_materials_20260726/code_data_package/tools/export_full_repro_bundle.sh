#!/usr/bin/env bash
# Export the exact custom checkpoints, source snapshot, and machine environment
# needed for end-to-end InTraScale inference reproduction.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/export_full_repro_bundle.sh --project-root PATH --output PATH

Optional environment variables:
  LIGHTX2V_REPO     LightX2V checkout used by the experiments
  DIFFSYNTH_REPO    DiffSynth-Studio checkout used for training
  VBENCH_REPO       VBench checkout used for evaluation
  VBENCH_PYTHON     Python executable of the separate VBench environment
  REALESRGAN_REPO   Real-ESRGAN checkout used by the RGB baseline
  MODEL_ROOT        Wan2.1-T2V-1.3B public model snapshot
  DISTILL_MODEL_ROOT 14B StepDistill-CfgDistill public model snapshot

The output directory must not already exist. The script never deletes source
files. Every custom checkpoint is checked against the paper run-manifest size
and SHA-256 before it is copied.
EOF
}

PROJECT_ROOT=""
OUTPUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --output)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PROJECT_ROOT}" || -z "${OUTPUT_ROOT}" ]]; then
  usage >&2
  exit 2
fi

PROJECT_ROOT="$(realpath "${PROJECT_ROOT}")"
if [[ ! -d "${PROJECT_ROOT}/paper/aaai27/experiments" ]]; then
  echo "Not a wanUpsampler checkout: ${PROJECT_ROOT}" >&2
  exit 1
fi

TOOLS_ROOT="${PROJECT_ROOT}/paper/aaai27/submission_materials_20260726/code_data_package"
ASSET_MANIFEST="${TOOLS_ROOT}/reproduction_assets.json"
if [[ ! -f "${ASSET_MANIFEST}" ]]; then
  echo "Missing asset manifest: ${ASSET_MANIFEST}" >&2
  exit 1
fi

OUTPUT_PARENT="$(realpath -m "$(dirname "${OUTPUT_ROOT}")")"
OUTPUT_ROOT="${OUTPUT_PARENT}/$(basename "${OUTPUT_ROOT}")"
if [[ "${OUTPUT_ROOT}" == "/" || "${OUTPUT_ROOT}" == "${PROJECT_ROOT}" ]]; then
  echo "Unsafe output path: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output already exists; choose a new directory: ${OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/checkpoints/custom" \
  "${OUTPUT_ROOT}/environment" \
  "${OUTPUT_ROOT}/git" \
  "${OUTPUT_ROOT}/source" \
  "${OUTPUT_ROOT}/public_models"

CHECKPOINT_ROWS="${OUTPUT_ROOT}/environment/custom_checkpoint_rows.tsv"
python3 - "${ASSET_MANIFEST}" > "${CHECKPOINT_ROWS}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for name, item in manifest["custom_checkpoints"].items():
    print(
        name,
        item["source_relative_to_project"],
        item["bundle_name"],
        item["size_bytes"],
        item["sha256"],
        sep="\t",
    )
PY

while IFS=$'\t' read -r logical_name relative_source bundle_name expected_size expected_sha; do
  source_path="${PROJECT_ROOT}/${relative_source}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing custom checkpoint ${logical_name}: ${source_path}" >&2
    exit 1
  fi
  actual_size="$(stat -c '%s' "${source_path}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "Size mismatch for ${logical_name}: expected ${expected_size}, got ${actual_size}" >&2
    exit 1
  fi
  actual_sha="$(sha256sum "${source_path}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "SHA-256 mismatch for ${logical_name}: expected ${expected_sha}, got ${actual_sha}" >&2
    exit 1
  fi
  cp --reflink=auto --preserve=timestamps "${source_path}" \
    "${OUTPUT_ROOT}/checkpoints/custom/${bundle_name}"
  printf '%s  %s\n' "${actual_sha}" "${bundle_name}" \
    >> "${OUTPUT_ROOT}/checkpoints/custom/SHA256SUMS"
  echo "exported ${logical_name}: ${bundle_name}"
done < "${CHECKPOINT_ROWS}"

# Capture the current code used by the method, training, inference, and paper
# evaluation paths. Large outputs, datasets, videos, and caches are excluded.
tar \
  --exclude='*.pyc' \
  --exclude='*/__pycache__' \
  --exclude='*/.pytest_cache' \
  --exclude='*/outputs' \
  --exclude='*/results' \
  --exclude='*/rewrite' \
  --exclude='*.mp4' \
  -C "${PROJECT_ROOT}" \
  -cf - \
  wan_sr \
  changing_resolution \
  changing_resolution_distill \
  paper/aaai27/experiments \
  paper/aaai27/submission_materials_20260726/code_data_package \
  | tar -C "${OUTPUT_ROOT}/source" -xf -

capture_git() {
  local name="$1"
  local repo="$2"
  if [[ -z "${repo}" || ! -d "${repo}/.git" ]]; then
    printf 'not_available\n' > "${OUTPUT_ROOT}/git/${name}.status"
    return
  fi
  git -C "${repo}" rev-parse HEAD > "${OUTPUT_ROOT}/git/${name}.commit"
  git -C "${repo}" status --porcelain=v2 > "${OUTPUT_ROOT}/git/${name}.status"
  git -C "${repo}" diff --binary HEAD > "${OUTPUT_ROOT}/git/${name}.patch"
  git -C "${repo}" remote -v > "${OUTPUT_ROOT}/git/${name}.remotes"
  git -C "${repo}" submodule status --recursive \
    > "${OUTPUT_ROOT}/git/${name}.submodules" 2>&1 || true
}

capture_git "wanupsampler" "${PROJECT_ROOT}"
capture_git "lightx2v" "${LIGHTX2V_REPO:-}"
capture_git "diffsynth" "${DIFFSYNTH_REPO:-}"
capture_git "vbench" "${VBENCH_REPO:-}"
capture_git "realesrgan" "${REALESRGAN_REPO:-}"

{
  date --iso-8601=seconds
  uname -a
  if [[ -f /etc/os-release ]]; then cat /etc/os-release; fi
} > "${OUTPUT_ROOT}/environment/os.txt"

{
  command -v lscpu >/dev/null 2>&1 && lscpu
  command -v free >/dev/null 2>&1 && free -h
} > "${OUTPUT_ROOT}/environment/cpu_memory.txt"

{
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi
  command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
      --format=csv,noheader
} > "${OUTPUT_ROOT}/environment/nvidia_smi.txt"

{
  command -v nvcc >/dev/null 2>&1 && nvcc --version
  command -v ffmpeg >/dev/null 2>&1 && ffmpeg -version
  command -v gcc >/dev/null 2>&1 && gcc --version
} > "${OUTPUT_ROOT}/environment/toolchain.txt"

python3 -VV > "${OUTPUT_ROOT}/environment/python.txt" 2>&1
python3 -m pip freeze --all > "${OUTPUT_ROOT}/environment/pip_freeze.txt"
if command -v conda >/dev/null 2>&1; then
  conda env export > "${OUTPUT_ROOT}/environment/conda_environment.yml"
  conda list --explicit > "${OUTPUT_ROOT}/environment/conda_explicit.txt"
fi
if [[ -n "${VBENCH_PYTHON:-}" ]]; then
  "${VBENCH_PYTHON}" -VV \
    > "${OUTPUT_ROOT}/environment/vbench_python.txt" 2>&1
  "${VBENCH_PYTHON}" -m pip freeze --all \
    > "${OUTPUT_ROOT}/environment/vbench_pip_freeze.txt"
fi

python3 - > "${OUTPUT_ROOT}/environment/python_runtime.json" <<'PY'
import importlib
import json
import os
import platform
import sys

modules = [
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
    "accelerate",
    "safetensors",
    "numpy",
    "scipy",
    "cv2",
    "lpips",
    "lightx2v",
    "lightx2v_platform",
]
result = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "modules": {},
}
for name in modules:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:
        result["modules"][name] = {"error": repr(exc)}
if "torch" in sys.modules:
    import torch

    result["torch_runtime"] = {
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY

for model_entry in \
  "wan13b:${MODEL_ROOT:-}" \
  "distill14b:${DISTILL_MODEL_ROOT:-}"; do
  model_name="${model_entry%%:*}"
  model_path="${model_entry#*:}"
  if [[ -n "${model_path}" && -d "${model_path}" ]]; then
    {
      echo "path=${model_path}"
      find "${model_path}" -maxdepth 3 -type f -printf '%P\t%s\n' | sort
    } > "${OUTPUT_ROOT}/public_models/${model_name}_inventory.tsv"
  else
    echo "not_available" > "${OUTPUT_ROOT}/public_models/${model_name}_inventory.tsv"
  fi
done

cp "${ASSET_MANIFEST}" "${OUTPUT_ROOT}/reproduction_assets.json"
(
  cd "${OUTPUT_ROOT}"
  find . -type f ! -name 'BUNDLE_SHA256SUMS' -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > BUNDLE_SHA256SUMS
)

echo "bundle=${OUTPUT_ROOT}"
echo "Run tools/verify_repro_bundle.py --bundle-root '${OUTPUT_ROOT}' --require-checkpoints"
