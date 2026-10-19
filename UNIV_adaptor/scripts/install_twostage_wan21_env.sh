#!/usr/bin/env bash
# One-time dependency installation for the two-stage platform on a NEW GPU machine.
#
#   bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh            # all steps
#   bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh <step>     # one step
#   steps: preflight wan vbench flash rife verify | cache-export (run on a WORKING machine)
#
# Idempotent: every step checks first and installs only what is missing. Shared
# environments under the repo (/mnt/afs_2) are reused when they work on this
# machine; when one does not, a new GPU-type-specific one is built next to it and
# never over the old one. Working interpreters are written to
# outputs/twostage_env/<gpu-tag>.env, which run_univ_twostage_wan21_8gpu.sh sources.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STEP="${1:-all}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT:-fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
CACHE_SEED="${CACHE_SEED:-/mnt/afs_2/houze/cache_seed}"
NGPUS="${NGPUS:-8}"
FLASH_ROOT="${REPO_ROOT}/UNIV_adaptor/external/flashvsr"
FLASH_KERNEL_ROOT="${REPO_ROOT}/UNIV_adaptor/external/flashvsr_block_sparse"
FLASH_WEIGHTS="${REPO_ROOT}/checkpoints/flashvsr_v11"
RIFE_ROOT="${REPO_ROOT}/UNIV_adaptor/external/practical_rife"
RIFE_WEIGHTS="${REPO_ROOT}/checkpoints/rife_v426"
RIFE_COMMIT="bbfd2ea90910789a860ea3e2b32a240cd577b75e"
RIFE_REVISION="01fdc7e97404120c243c3ea7b427046e5dc7643e"
RIFE_ZIP_SHA256="1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"

say() { printf '\n==== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
gpu_tag() { nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9'; }
ENV_DIR="${REPO_ROOT}/outputs/twostage_env"
ENV_FILE=""

record_env() {  # record_env KEY VALUE: persist one working interpreter for the run script
  mkdir -p "${ENV_DIR}"
  touch "${ENV_FILE}"
  grep -v "^export $1=" "${ENV_FILE}" > "${ENV_FILE}.tmp" || true
  echo "export $1=\"$2\"" >> "${ENV_FILE}.tmp"
  mv "${ENV_FILE}.tmp" "${ENV_FILE}"
}

current() {  # current KEY DEFAULT: interpreter from the env file, else default
  local value=""
  [[ -f "${ENV_FILE}" ]] && value="$(sed -n "s/^export $1=\"\(.*\)\"$/\1/p" "${ENV_FILE}")"
  echo "${value:-$2}"
}

preflight() {
  say "preflight"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  local count; count="$(nvidia-smi -L | wc -l)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  (( count >= NGPUS )) || die "Found ${count} GPUs, NGPUS=${NGPUS}"
  for tool in git curl unzip sha256sum; do command -v "${tool}" >/dev/null || die "Missing ${tool} (apt-get install -y ${tool})"; done
  for tool in ffmpeg ffprobe; do command -v "${tool}" >/dev/null || die "Missing ${tool}: apt-get install -y ffmpeg (or conda install -c conda-forge ffmpeg)"; done
  [[ -x "${BASE_PYTHON}" ]] || die "BASE_PYTHON not found: ${BASE_PYTHON}"
  "${BASE_PYTHON}" -c 'import torch, sys; assert torch.cuda.is_available(), "no CUDA"; print("base", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)' \
    || die "BASE_PYTHON must be the working CUDA PyTorch image (Wan venv inherits its torch/flash-attn)"
  for f in config.json Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth; do
    [[ -f "${MODEL_ROOT}/${f}" ]] || die "Wan2.1-T2V-1.3B weights incomplete: ${MODEL_ROOT}/${f}"
  done
  local free_gb; free_gb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print int($4/1024/1024)}')"
  echo "Free space under repo: ${free_gb} GB (run needs ~40 GB: lossless stage-1 frames + videos)"
  (( free_gb >= 40 )) || die "Less than 40 GB free under ${REPO_ROOT}"
  echo "Env file for this GPU type: ${ENV_FILE}"
}

wan() {
  say "Wan2.1 generation environment (stage 1, RIFE, controls)"
  "${BASE_PYTHON}" UNIV_adaptor/scripts/data/fetch_published_accelerators.py --fetch wan21
  local venv python
  venv="${REPO_ROOT}/.venv_published_wan21"
  python="$(current WAN_PYTHON "${venv}/bin/python")"
  local probe='import torch, flash_attn, diffusers, transformers, imageio, imageio_ffmpeg, cv2, easydict, ftfy, PIL, einops, numpy
assert torch.cuda.is_available()
from flash_attn import flash_attn_varlen_func
q = torch.randn(17, 12, 128, device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, q, q, cu, cu, 17, 17)
torch.cuda.synchronize(); assert out.shape == q.shape
print("wan env OK: torch", torch.__version__, "flash_attn", flash_attn.__version__)'
  if [[ -x "${python}" ]] && "${python}" -c "${probe}"; then
    record_env WAN_PYTHON "${python}"
    return
  fi
  if [[ -x "${python}" ]] && "${python}" -c 'import torch, flash_attn; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "Python-only packages missing; installing the pinned requirements into ${python}"
    "${python}" -m pip install -r UNIV_adaptor/configs/published_wan21_requirements.txt
  else
    venv="${REPO_ROOT}/.venv_published_wan21_$(gpu_tag)"
    echo "Shared Wan venv does not run here; building ${venv} (system site packages from ${BASE_PYTHON})"
    PILOT_VENV="${venv}" BASE_PYTHON="${BASE_PYTHON}" INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}" \
      bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh setup
    python="${venv}/bin/python"
  fi
  "${python}" -c "${probe}" || die "Wan env still broken; if flash-attn is missing rerun with INSTALL_FLASH_ATTN=1"
  record_env WAN_PYTHON "${python}"
}

cache_export() {
  say "Export VBench model caches from THIS (working) machine to ${CACHE_SEED}"
  mkdir -p "${CACHE_SEED}"
  for d in vbench clip torch/hub; do
    if [[ -d "${HOME}/.cache/${d}" ]]; then
      mkdir -p "${CACHE_SEED}/${d}"
      cp -a "${HOME}/.cache/${d}/." "${CACHE_SEED}/${d}/"
      echo "copied ~/.cache/${d}"
    fi
  done
  du -sh "${CACHE_SEED}"
}

vbench() {
  say "VBench scoring environment (${VBENCH_PYTHON})"
  [[ -f "${VBENCH_ROOT}/evaluate.py" ]] || die "VBench checkout missing: ${VBENCH_ROOT}"
  [[ "$(git -C "${VBENCH_ROOT}" rev-parse HEAD)" == "${EXPECTED_VBENCH_COMMIT}" ]] || die "VBench is not at ${EXPECTED_VBENCH_COMMIT}"
  # Never let pip replace the image's CUDA torch stack.
  local pins; pins="$(mktemp)"
  "${VBENCH_PYTHON}" -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy)==' > "${pins}" || true
  echo "pip constraints:"; cat "${pins}"
  declare -A package=([clip]="git+https://github.com/openai/CLIP.git" [omegaconf]=omegaconf [easydict]=easydict
    [timm]=timm [pyiqa]=pyiqa [decord]=decord [cv2]=opencv-python-headless [scipy]=scipy [transformers]=transformers
    [PIL]=pillow [yaml]=pyyaml [einops]=einops [ftfy]=ftfy [regex]=regex [tqdm]=tqdm)
  local missing=()
  for module in ftfy regex clip omegaconf easydict timm pyiqa decord cv2 scipy transformers PIL yaml einops tqdm; do
    "${VBENCH_PYTHON}" -c "import ${module}" 2>/dev/null || missing+=("${package[${module}]}")
  done
  if (( ${#missing[@]} )); then
    echo "Installing missing: ${missing[*]}"
    "${VBENCH_PYTHON}" -m pip install -c "${pins}" "${missing[@]}"
  fi
  (cd "${VBENCH_ROOT}" && "${VBENCH_PYTHON}" -c 'import torch, clip, omegaconf, easydict, timm, pyiqa, decord, cv2, scipy, transformers
assert torch.cuda.is_available(); print("vbench imports OK: torch", torch.__version__)')
  # Model caches: seed from shared storage when present (see cache-export), then fill gaps.
  for d in vbench clip torch/hub; do
    if [[ -d "${CACHE_SEED}/${d}" ]]; then
      mkdir -p "${HOME}/.cache/${d}"
      cp -an "${CACHE_SEED}/${d}/." "${HOME}/.cache/${d}/"
      echo "seeded ~/.cache/${d} from ${CACHE_SEED}"
    fi
  done
  # torch.hub.load('facebookresearch/dino:main') hits the rate-limited GitHub API
  # unless the repo is already in the hub cache.
  local hub="${HOME}/.cache/torch/hub"
  mkdir -p "${hub}/checkpoints"
  if [[ ! -f "${hub}/facebookresearch_dino_main/vision_transformer.py" ]]; then
    rm -rf "${hub}/facebookresearch_dino_main"
    git clone --depth 1 https://github.com/facebookresearch/dino.git "${hub}/facebookresearch_dino_main"
  fi
  for w in dino_vitbase16_pretrain dino_deitsmall16_pretrain; do
    [[ -s "${hub}/checkpoints/${w}.pth" ]] || curl -fL --retry 5 -o "${hub}/checkpoints/${w}.pth" "https://dl.fbaipublicfiles.com/dino/${w}/${w}.pth"
  done
  "${VBENCH_PYTHON}" -c 'import clip
for name in ("ViT-B/32", "ViT-L/14"):
    clip.load(name, device="cpu"); print("CLIP", name, "cached")'
  echo "Remaining VBench models (ViCLIP, AMT, MUSIQ, RAFT, aesthetic) download on first use; verify runs a 1-GPU smoke for all 7 dimensions."
}

flash() {
  say "FlashVSR environment (isolated torch 2.6 + Block-Sparse-Attention)"
  if [[ ! -d "${FLASH_ROOT}" || ! -d "${FLASH_KERNEL_ROOT}" ]]; then
    BASE_PYTHON="${BASE_PYTHON}" bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh fetch
  fi
  local venv python
  venv="${REPO_ROOT}/.venv_flashvsr_diagnostic"
  python="$(current FLASH_PYTHON "${venv}/bin/python")"
  local probe='import torch, imageio, einops, safetensors
from block_sparse_attn import block_sparse_attn_func
assert torch.cuda.is_available(); torch.zeros(1, device="cuda")
major, minor = torch.cuda.get_device_capability(); assert (major, minor) in ((8, 0), (9, 0)), f"block-sparse build has no sm_{major}{minor}"
print("flash env OK: torch", torch.__version__, "cuda", torch.version.cuda, "sm", major, minor)'
  if ! { [[ -x "${python}" ]] && "${python}" -c "${probe}"; }; then
    venv="${REPO_ROOT}/.venv_flashvsr_$(gpu_tag)"
    python="${venv}/bin/python"
    echo "Shared FlashVSR venv does not run here; building ${venv}"
    command -v nvcc >/dev/null || [[ -x /usr/local/cuda/bin/nvcc ]] || die "nvcc (CUDA toolkit >= 12.4) is required to compile Block-Sparse-Attention"
    export PATH="${PATH}:/usr/local/cuda/bin"
    local py311="${FLASH_BASE_PYTHON:-}"
    if [[ -z "${py311}" ]]; then
      if "${BASE_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
        py311="${BASE_PYTHON}"
      else
        local conda; conda="$(dirname "${BASE_PYTHON}")/conda"
        [[ -x "${conda}" ]] || die "Need Python 3.11: set FLASH_BASE_PYTHON"
        [[ -x "${REPO_ROOT}/.conda_py311/bin/python" ]] || "${conda}" create -y -p "${REPO_ROOT}/.conda_py311" python=3.11
        py311="${REPO_ROOT}/.conda_py311/bin/python"
      fi
    fi
    FLASH_VENV="${venv}" BASE_PYTHON="${py311}" BLOCK_SPARSE_ATTN_CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-80;90}" \
      MAX_JOBS="${MAX_JOBS:-4}" bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh setup
    "${python}" -c "${probe}" || die "FlashVSR env still broken"
  fi
  # Verifies an existing SHA256-bound download, or downloads the pinned revision.
  FLASH_VENV="${venv}" FLASH_PYTHON="${python}" bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh download
  record_env FLASH_PYTHON "${python}"
}

rife() {
  say "Practical-RIFE v4.26 (runs in the Wan env)"
  if [[ ! -d "${RIFE_ROOT}" ]]; then
    git clone --no-checkout https://github.com/hzwer/Practical-RIFE.git "${RIFE_ROOT}"
    git -C "${RIFE_ROOT}" checkout --detach "${RIFE_COMMIT}"
  fi
  [[ "$(git -C "${RIFE_ROOT}" rev-parse HEAD)" == "${RIFE_COMMIT}" ]] || die "Practical-RIFE is not at ${RIFE_COMMIT}"
  [[ -z "$(git -C "${RIFE_ROOT}" status --porcelain --untracked-files=all)" ]] || die "Practical-RIFE checkout is dirty"
  if [[ -f "${RIFE_WEIGHTS}/rife_weight_binding.json" ]]; then
    "${BASE_PYTHON}" -c 'import json, sys
sys.path.insert(0, "UNIV_adaptor/scripts/data")
import flashvsr_asset_diagnostic as d
d.verify_files(json.load(open(sys.argv[1]))["files"]); print("RIFE weights verified")' "${RIFE_WEIGHTS}/rife_weight_binding.json"
    return
  fi
  # Author's own Hugging Face copy of the v4.26 release (the README links Google Drive).
  local download="${RIFE_WEIGHTS}/download" zip="RIFEv4.26_0921.zip"
  mkdir -p "${download}"
  [[ -s "${download}/${zip}" ]] || curl -fL --retry 5 -o "${download}/${zip}" "${HF_ENDPOINT}/hzwer/RIFE/resolve/${RIFE_REVISION}/${zip}"
  echo "${RIFE_ZIP_SHA256}  ${download}/${zip}" | sha256sum -c - || die "RIFE archive hash mismatch"
  rm -rf "${download}/x" && unzip -q "${download}/${zip}" -d "${download}/x"
  mkdir -p "${RIFE_WEIGHTS}/train_log"
  for f in flownet.pkl IFNet_HDv3.py RIFE_HDv3.py refine.py; do cp "${download}/x/RIFEv4.26_0921/${f}" "${RIFE_WEIGHTS}/train_log/${f}"; done
  rm -rf "${download}/x"
  "${BASE_PYTHON}" - "${RIFE_WEIGHTS}" "${RIFE_REVISION}" "${RIFE_ZIP_SHA256}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "UNIV_adaptor/scripts/data")
import flashvsr_asset_diagnostic as d
root, revision, archive = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
files = [d.bound_file(root / "train_log" / n) for n in ("flownet.pkl", "IFNet_HDv3.py", "RIFE_HDv3.py", "refine.py")]
d.write_new(root / "rife_weight_binding.json", {"repository": "hzwer/RIFE", "revision": revision,
    "archive": "RIFEv4.26_0921.zip", "archive_sha256": archive, "model_parent": str(root), "files": files})
print("RIFE weights extracted and bound:", root)
PY
}

verify() {
  say "verify: real full-shape runs on this machine"
  local smoke="${SMOKE_VIDEO:-}"
  if [[ -z "${smoke}" ]]; then
    smoke="$(find "${REPO_ROOT}/outputs" -path '*videos/pilot/FULL50/*.mp4' -print -quit 2>/dev/null || true)"
  fi
  [[ -n "${smoke}" ]] || die "Set SMOKE_VIDEO to any existing 832x480 MP4 for the VBench smoke test"
  TWOSTAGE_ENV_FILE="${ENV_FILE}" SMOKE_VIDEO="${smoke}" VBENCH_PYTHON="${VBENCH_PYTHON}" \
    bash UNIV_adaptor/scripts/run_univ_twostage_wan21_8gpu.sh vbench-smoke
  # Freezes the plan, then: every Wan canvas, FlashVSR (sparse + color fix), RIFE, controls.
  TWOSTAGE_ENV_FILE="${ENV_FILE}" bash UNIV_adaptor/scripts/run_univ_twostage_wan21_8gpu.sh check
  echo "Installation verified. Next: bash UNIV_adaptor/scripts/run_univ_twostage_wan21_8gpu.sh all"
}

ENV_FILE="${TWOSTAGE_ENV_FILE:-${ENV_DIR}/$(gpu_tag).env}"
case "${STEP}" in
  all) preflight; wan; vbench; flash; rife; verify ;;
  preflight|wan|vbench|flash|rife|verify) "${STEP}" ;;
  cache-export) cache_export ;;
  *) die "Unknown step: ${STEP}" ;;
esac
