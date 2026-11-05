#!/usr/bin/env bash
# One-time build of the attention-backend environment for attention_arms.py (SAGE2, SPARGE, SVG2).
#
#   bash UNIV_adaptor/scripts/install_attention_env.sh            # all steps
#   bash UNIV_adaptor/scripts/install_attention_env.sh <step>     # one step
#   steps: preflight venv sage sparge flashinfer verify probe
#
# The venv lives on the shared afs repo and is built once per GPU type (.venv_attn_<gpu-tag>), inheriting the base
# image's torch / flash-attn / LightX2V dependencies (--system-site-packages); the compiled kernels go only into
# the venv, never into /opt/conda. Another machine with the same GPU type and image reuses it as is. Idempotent:
# each step first checks whether its library already works and builds only if not. Sources are cloned into
# UNIV_adaptor/external/ and the commit of each build is recorded in outputs/attention_env/<gpu-tag>.env, which also
# holds ATTN_PYTHON for the probe and the run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STEP="${1:-all}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
LIGHTX2V="${LIGHTX2V:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
EXT="${REPO_ROOT}/UNIV_adaptor/external"
MAX_JOBS="${MAX_JOBS:-32}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MAX_JOBS
cd "${REPO_ROOT}"

say() { printf '\n==== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
gpu_tag() { nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9'; }
TAG="$(gpu_tag)"
ENV_DIR="${REPO_ROOT}/outputs/attention_env"
ENV_FILE="${ATTN_ENV_FILE:-${ENV_DIR}/${TAG}.env}"
VENV="${REPO_ROOT}/.venv_attn_${TAG}"
PY="${VENV}/bin/python"

record_env() {  # record_env KEY VALUE
  mkdir -p "${ENV_DIR}"
  touch "${ENV_FILE}"
  grep -v "^export $1=" "${ENV_FILE}" > "${ENV_FILE}.tmp" || true
  echo "export $1=\"$2\"" >> "${ENV_FILE}.tmp"
  mv "${ENV_FILE}.tmp" "${ENV_FILE}"
}

arch() {  # compute capability of GPU 0, e.g. 9.0
  "${BASE_PYTHON}" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))'
}

clone() {  # clone URL DIR: shallow clone with submodules, once
  [[ -d "$2/.git" ]] || git clone --recursive "$1" "$2"
  git -C "$2" submodule update --init --recursive
  git -C "$2" rev-parse HEAD
}

check() {  # check NAME PYTHON_SNIPPET: run on GPU in the venv
  [[ -x "${PY}" ]] && "${PY}" -c "$2" > /tmp/attn_check_$1.log 2>&1 && { echo "  OK    $1: $(tail -1 /tmp/attn_check_$1.log)"; return 0; }
  echo "  MISS  $1 ($(tail -1 /tmp/attn_check_$1.log 2>/dev/null || echo 'venv missing'))"
  return 1
}

QKV='import torch
q, k, v = (torch.randn(1, 4096, 12, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))'
SAGE_OK="${QKV}
from sageattention import sageattn
o = sageattn(q, k, v, tensor_layout=\"NHD\"); ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
print('sageattention', __import__('sageattention').__file__, 'max err', float((o - ref).abs().max()))"
SPARGE_OK='import spas_sage_attn._fused, spas_sage_attn._qattn
from spas_sage_attn.utils import get_vanilla_qk_quant
print("spas_sage_attn", spas_sage_attn._fused.__file__)'
FLASHINFER_OK='import flashinfer
assert hasattr(flashinfer.sparse, "VariableBlockSparseAttentionWrapper")
print("flashinfer", flashinfer.__version__)'

preflight() {
  say "preflight (read-only)"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo "GPU tag ${TAG}, compute capability $(arch)"
  case "$(arch)" in 8.0|8.6|8.9|9.0) ;; *) echo "  NOTE  untested compute capability $(arch): SageAttention2 / SpargeAttn kernels exist for sm80/86/89/90" ;; esac
  "${BASE_PYTHON}" -c 'import torch, sys; assert torch.cuda.is_available(); print("base python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)' \
    || die "BASE_PYTHON must be the CUDA PyTorch image: ${BASE_PYTHON}"
  "${BASE_PYTHON}" -c 'import flash_attn; print("flash_attn", flash_attn.__version__)' || die "flash_attn missing in the base image (FULL50_LX uses it)"
  command -v nvcc >/dev/null || [[ -x /usr/local/cuda/bin/nvcc ]] || die "nvcc (CUDA toolkit) is required to compile the kernels"
  export PATH="${PATH}:/usr/local/cuda/bin"; echo "nvcc: $(nvcc --version | tail -1)"
  [[ -d "${LIGHTX2V}/lightx2v" ]] || die "LightX2V not found at ${LIGHTX2V}"
  PYTHONPATH="${LIGHTX2V}" "${BASE_PYTHON}" -c 'import lightx2v.infer' > /tmp/attn_lx2v.log 2>&1 \
    && echo "LightX2V importable from the base python" || die "LightX2V does not import from ${BASE_PYTHON}: $(tail -1 /tmp/attn_lx2v.log)"
  [[ -f "${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth" ]] || die "Wan2.1-T2V-1.3B weights not found under ${MODEL_ROOT}"
  git ls-remote https://github.com/thu-ml/SageAttention HEAD >/dev/null 2>&1 && echo "GitHub reachable" \
    || echo "  NOTE  GitHub not reachable: set a proxy, or pre-clone the sources into ${EXT}"
  echo "venv: ${VENV}; env file: ${ENV_FILE}"
}

venv() {
  say "venv ${VENV} (inherits the base image's site-packages)"
  [[ -x "${PY}" ]] || "${BASE_PYTHON}" -m venv --system-site-packages "${VENV}"
  "${PY}" -m pip install -q --upgrade pip wheel setuptools ninja packaging
  "${PY}" -c 'import torch; assert torch.cuda.is_available(); print("venv torch", torch.__version__)'
  record_env ATTN_PYTHON "${PY}"
}

sage() {
  say "SageAttention2 (SAGE2; also the kernel library SPARGE builds on)"
  check sage "${SAGE_OK}" && return 0
  local commit; commit="$(clone https://github.com/thu-ml/SageAttention "${EXT}/SageAttention" | tail -1)"
  ( cd "${EXT}/SageAttention" && CUDA_ARCHITECTURES="$(arch)" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" \
      "${PY}" -m pip install -v --no-build-isolation . )
  check sage "${SAGE_OK}" || die "SageAttention built but does not run (log /tmp/attn_check_sage.log)"
  record_env SAGEATTENTION_COMMIT "${commit}"
}

sparge() {
  say "SpargeAttn kernels (spas_sage_attn; SPARGE)"
  check sparge "${SPARGE_OK}" && return 0
  local commit; commit="$(clone https://github.com/thu-ml/SpargeAttn "${EXT}/SpargeAttn" | tail -1)"
  ( cd "${EXT}/SpargeAttn" && TORCH_CUDA_ARCH_LIST="$(arch)" "${PY}" -m pip install -v --no-build-isolation . )
  check sparge "${SPARGE_OK}" || die "spas_sage_attn built but does not import (log /tmp/attn_check_sparge.log)"
  record_env SPARGEATTN_COMMIT "${commit}"
}

flashinfer() {
  say "FlashInfer (SVG2's variable-block sparse attention)"
  check flashinfer "${FLASHINFER_OK}" && return 0
  "${PY}" -m pip install flashinfer-python
  check flashinfer "${FLASHINFER_OK}" || die "flashinfer installed but lacks sparse.VariableBlockSparseAttentionWrapper; \
see github.com/svg-project/Sparse-VideoGen for the flashinfer build SVG2 expects (log /tmp/attn_check_flashinfer.log)"
  record_env FLASHINFER_VERSION "$("${PY}" -c 'import flashinfer; print(flashinfer.__version__)')"
}

verify() {
  say "verify: every backend in the venv"
  local bad=0
  check sage "${SAGE_OK}" || bad=1
  check sparge "${SPARGE_OK}" || bad=1
  check flashinfer "${FLASHINFER_OK}" || bad=1
  (( bad == 0 )) && echo "All three backends work. Next: bash $0 probe" || echo "Some backends are missing; rerun their step to see the build error."
  return "${bad}"
}

probe() {
  say "probe: one 832x480x81 50-step video per arm through LightX2V (~10 min on 4 GPUs)"
  "${BASE_PYTHON}" UNIV_adaptor/scripts/data/attention_arms.py probe --out outputs/attention_probe --python "${PY}" \
    --lightx2v "${LIGHTX2V}" --model "${MODEL_ROOT}" --gpus "${GPUS:-0,1,2,3}"
  cp "${ENV_FILE}" outputs/attention_probe/ 2>/dev/null || true
  tar czf outputs/attention_probe.tgz -C outputs attention_probe
  echo "SEND THIS: ${REPO_ROOT}/outputs/attention_probe.tgz"
}

case "${STEP}" in
  all)
    preflight; venv
    # A failed build stops only its own step (subshell); the probe then reports that arm as failed.
    for s in sage sparge flashinfer; do ( "${s}" ) || echo "STEP FAILED: ${s} (rerun: bash $0 ${s})"; done
    verify || true
    probe ;;
  preflight|venv|sage|sparge|flashinfer|verify|probe) "${STEP}" ;;
  *) die "Unknown step: ${STEP}" ;;
esac
