#!/usr/bin/env bash
# Natural prompt set v1 (natural_prompts.py): 256 real VidProM user prompts x 1 seed x 8 frontier arms on one 8-GPU
# machine, scored by the same three layers, then prompt-vector routing vs blind / predicted cells.
# On a new machine first: bash UNIV_adaptor/scripts/run_univ_natural_8gpu.sh env   # read-only readiness check
# Two steps, because the prompt set is reviewed and committed before any GPU time is spent on it:
#   bash UNIV_adaptor/scripts/run_univ_natural_8gpu.sh prompts   # ~20 min: download, filter, Qwen screen, draw 256
#        -> send outputs/natural_v1_prompts.tgz; the prompts and configs come back as a commit; git pull
#   bash UNIV_adaptor/scripts/run_univ_natural_8gpu.sh all       # ~9 h: everything below, in this order
# Stages (each resumes from its own receipts): checklist platform twostage fixed dynamic quality route; status.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-help}"
OUT="${REPO_ROOT}/outputs"
S="${REPO_ROOT}/UNIV_adaptor/scripts"
CFG="${REPO_ROOT}/UNIV_adaptor/configs"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-${REPO_ROOT}/.venv_vlm3/bin/python}"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/afs_2/houze/models/Qwen3-VL-32B-Instruct}"
PROMPTS="${REPO_ROOT}/prompts/univ_natural_v1.jsonl"
CONTROLLED=("${REPO_ROOT}/prompts/univ_controlled_factor_v1.jsonl" "${REPO_ROOT}/prompts/univ_controlled_factor_dev2.jsonl")
SEL="${OUT}/natural_v1_prompts"
P="${OUT}/published_wan21_platform_natural"
T="${OUT}/published_wan21_twostage_natural"
GEN="${OUT}/requirements_natural"
CHECKLIST="${GEN}/checklist.jsonl"
FIXED="${OUT}/fixed_metrics_natural"
DYN="${OUT}/dynamic_natural"
Q="${OUT}/quality_natural"
ROUTE="${OUT}/route_natural"
ALL_GPUS="0,1,2,3,4,5,6,7"
IFS=',' read -r -a GPUS <<< "${ALL_GPUS}"
export NGPUS=8 PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"
say() { printf '\n######## %s  (%s)\n' "$*" "$(date '+%F %T')"; }
need() { [[ -f "$1" ]] || { echo "missing $1 ($2)" >&2; exit 1; }; }

gen_env() {  # interpreters verified by install_twostage_wan21_env.sh, as in the dev2 batch
  local tag env_file
  tag="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
  env_file="${TWOSTAGE_ENV_FILE:-${OUT}/twostage_env/${tag}.env}"
  need "${env_file}" "run install_twostage_wan21_env.sh first"
  # shellcheck disable=SC1090
  source "${env_file}"
  export TWOSTAGE_ENV_FILE="${env_file}" PYTHON_BIN="${WAN_PYTHON}" GEN_PYTHON="${WAN_PYTHON}"
  export VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
}

envcheck() {
  # Read-only: probes what every stage needs on THIS machine and names the fix for each failure.
  local bad=0 tag env_file
  ok() { printf '  OK    %s\n' "$1"; }
  no() { printf '  FAIL  %s\n        fix: %s\n' "$1" "$2"; bad=1; }
  try() { local what="$1" fix="$2"; shift 2; if "$@" > /tmp/natural_env_probe.log 2>&1; then ok "${what}"; else no "${what} ($(tail -1 /tmp/natural_env_probe.log))" "${fix}"; fi; }
  say "env: machine"
  local n; n="$(nvidia-smi -L 2>/dev/null | wc -l)"
  (( n >= 8 )) && ok "${n} GPUs: $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)" || no "${n} GPUs (need 8)" "use an 8-GPU machine or edit ALL_GPUS"
  for t in git curl unzip sha256sum ffmpeg ffprobe; do command -v "${t}" >/dev/null && ok "${t}" || no "${t}" "apt-get install -y ${t/ffprobe/ffmpeg}"; done
  local free; free="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print int($4/1024/1024)}')"
  (( free >= 80 )) && ok "${free} GB free under outputs" || no "${free} GB free under outputs (need ~80)" "free space on afs"
  try "Hugging Face reachable (${HF_ENDPOINT:-https://huggingface.co})" "set HF_ENDPOINT / proxy, or pre-download VidProM csv (see doc)" \
    curl -fsSI --max-time 20 "${HF_ENDPOINT:-https://huggingface.co}/api/datasets/WenhaoWang/VidProM"
  say "env: shared files on afs"
  local f
  for f in /mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth "${JUDGE_MODEL}/config.json" \
           /mnt/afs_2/houze/VBench/evaluate.py /mnt/afs_2/houze/LightX2V/lightx2v "${REPO_ROOT}/checkpoints/rcm/rCM_Wan2.1_T2V_1.3B_480p.pt" \
           "${REPO_ROOT}/checkpoints/rife_v426/rife_weight_binding.json" "${REPO_ROOT}/UNIV_adaptor/external/flashvsr" \
           "${OUT}/fixed_metrics_v1/fixed_by_video.csv"; do
    [[ -e "${f}" ]] && ok "${f}" || no "${f} missing" "bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh (rcm/rife/flash), or copy from the old machine"
  done
  say "env: base python ${BASE_PYTHON} (fixed layer, VBench, routing, T5)"
  try "torch + CUDA" "this image needs a CUDA PyTorch at ${BASE_PYTHON}" \
    "${BASE_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'
  try "numpy scipy cv2 PIL" "${BASE_PYTHON} -m pip install scipy opencv-python-headless pillow" \
    "${BASE_PYTHON}" -c 'import numpy, scipy.optimize, cv2, PIL'
  try "RAFT weights (torchvision; downloads once to ~/.cache)" "needs network to download.pytorch.org once, or copy ~/.cache/torch/hub/checkpoints" \
    "${BASE_PYTHON}" -c 'from torchvision.models.optical_flow import Raft_Large_Weights, raft_large; raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)'
  try "VBench imports" "bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh vbench" \
    "${BASE_PYTHON}" -c 'import clip, omegaconf, easydict, timm, pyiqa, decord'
  [[ -f "${HOME}/.cache/torch/hub/facebookresearch_dino_main/vision_transformer.py" ]] && ok "DINO in torch hub cache" \
    || no "DINO not in ~/.cache/torch/hub (VBench subject_consistency)" "bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh vbench"
  try "LightX2V importable (T5 embeddings)" "pip install LightX2V's requirements into ${BASE_PYTHON}, or set T5_PYTHON" \
    env PYTHONPATH="/mnt/afs_2/houze/LightX2V:${REPO_ROOT}" "${BASE_PYTHON}" -c 'import lightx2v, changing_resolution_uni.scripts.data.extract_prompt_t5_embeddings'
  say "env: judge python ${VLM_PYTHON} (screen, checklist, dynamic, quality, Qwen embeddings)"
  try "torch + CUDA + transformers>=4.57 + huggingface_hub + cv2" "build a venv for this machine next to the old one: ${BASE_PYTHON} -m venv --system-site-packages .venv_vlm3_new && .venv_vlm3_new/bin/pip install 'transformers>=4.57' 'accelerate>=1.0' huggingface_hub opencv-python-headless; then export VLM_PYTHON=\$PWD/.venv_vlm3_new/bin/python" \
    "${VLM_PYTHON}" -c 'import torch, transformers, huggingface_hub, cv2; assert torch.cuda.is_available(); from transformers import AutoModelForImageTextToText; v = tuple(map(int, transformers.__version__.split(".")[:2])); assert v >= (4, 57), transformers.__version__; print(transformers.__version__)'
  try "Qwen3-VL processor loads" "check ${JUDGE_MODEL}" \
    "${VLM_PYTHON}" -c "import transformers; transformers.AutoProcessor.from_pretrained('${JUDGE_MODEL}')"
  say "env: generation interpreters (install_twostage_wan21_env.sh)"
  tag="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
  env_file="${TWOSTAGE_ENV_FILE:-${OUT}/twostage_env/${tag}.env}"
  if [[ -f "${env_file}" ]]; then
    ok "${env_file}"
    # shellcheck disable=SC1090
    ( source "${env_file}"
      try "WAN_PYTHON runs flash-attn on this GPU" "bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh wan" \
        "${WAN_PYTHON}" -c 'import torch, flash_attn; assert torch.cuda.is_available()'
      try "FLASH_PYTHON runs block-sparse attention on this GPU" "bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh flash" \
        "${FLASH_PYTHON}" -c 'import torch; from block_sparse_attn import block_sparse_attn_func; torch.zeros(1, device="cuda")'
      exit "${bad}" ) || bad=1
  else
    no "no generation env file for GPU type '${tag}'" "bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh preflight && ... wan && ... vbench && ... flash && ... rife && ... rcm"
  fi
  say "env: $( (( bad )) && echo 'NOT READY - fix the FAIL lines, then rerun env' || echo 'READY')"
  return "${bad}"
}

prompts() {
  local d="${REPO_ROOT}/UNIV_adaptor/scripts/data/natural_prompts.py"
  say "prompts: VidProM candidates"
  [[ -f "${SEL}/candidates.jsonl" ]] || "${VLM_PYTHON}" "${d}" candidates --out "${SEL}"
  say "prompts: Qwen screen (one GPU)"
  [[ -f "${SEL}/screen.json" ]] || CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${VLM_PYTHON}" "${d}" screen --out "${SEL}" --model "${JUDGE_MODEL}"
  "${BASE_PYTHON}" "${d}" select --out "${SEL}" | tee "${SEL}/select.txt"
  "${BASE_PYTHON}" "${d}" configs --prompts "${SEL}/univ_natural_v1.jsonl" --config-dir "${SEL}"
  tar czf "${SEL}.tgz" -C "${OUT}" "$(basename "${SEL}")/univ_natural_v1.jsonl" "$(basename "${SEL}")/select.txt" \
    "$(basename "${SEL}")/candidates.jsonl" "$(basename "${SEL}")/candidates_meta.json" "$(basename "${SEL}")/screen.json" \
    "$(basename "${SEL}")"/published_wan21_*_natural.json
  echo "SEND THIS: ${SEL}.tgz"
}

checklist() {
  say "checklist: dynamic-layer questions for the natural prompts (Qwen3-VL-32B, machine only)"
  need "${PROMPTS}" "run the prompts stage, send the tgz, git pull"
  local ids pids=() i
  mapfile -t ids < <("${BASE_PYTHON}" -c 'import json,sys; [print(json.loads(l)["prompt_id"]) for l in open(sys.argv[1]) if l.strip()]' "${PROMPTS}")
  mkdir -p "${GEN}/logs"
  for i in "${!GPUS[@]}"; do
    # Few-shot examples are the pilot's watchmaking family, as for v2.0; its prompts live in the controlled files.
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${VLM_PYTHON}" "${S}/data/requirements_v2.py" generate \
      --prompts "${PROMPTS}" "${CONTROLLED[@]}" --examples "${REPO_ROOT}/prompts/univ_requirements_v2_pilot.jsonl" \
      --model "${JUDGE_MODEL}" --out "${GEN}" --only "${ids[@]}" --shard-index "${i}" --shard-count "${#GPUS[@]}" \
      > "${GEN}/logs/generate_${i}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "worker ${i} failed: ${GEN}/logs/generate_${i}.log" >&2; failed=1; }; done
  (( failed == 0 )) || return 1
  # Prompts whose checklist still fails validation are listed and left out of the dynamic layer.
  "${BASE_PYTHON}" "${S}/data/requirements_v2.py" collect --prompts "${PROMPTS}" --out "${GEN}" | tee "${GEN}/collect.txt"
}

platform() {
  say "platform natural: FULL50 STEP20 TEA015 SCALING10"
  gen_env
  local m
  for m in check calibrate generate finalize score export; do
    say "platform ${m}"
    PILOT_CONFIG="${CFG}/published_wan21_platform_natural.json" PUBLISHED_WAN21_ROOT="${P}" \
      bash "${S}/run_univ_published_wan21_8gpu.sh" "${m}"
  done
}

twostage() {
  say "two-stage natural: LR_FLASH F41_RIFE RCM4 LR41_FLASH_RIFE (FULL50 from platform)"
  gen_env
  bash "${S}/install_twostage_wan21_env.sh" rcm
  need "${P}/dataset_manifest.json" "platform natural must be finalized first"
  TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_natural.json" TWOSTAGE_ROOT="${T}" REFERENCE_ROOT="${P}" \
    bash "${S}/run_univ_twostage_wan21_8gpu.sh" all
}

fixed() {
  say "fixed layer"
  FIXED_ROOT="${FIXED}" PUBLISHED="${P} ${T}" DVG_STUDIES="" GPU_IDS="${ALL_GPUS}" bash "${S}/run_univ_fixed_metrics.sh" all
}

dynamic() {
  say "dynamic layer (retention)"
  need "${CHECKLIST}" "checklist stage"
  local m
  for m in plan run; do
    FIXED_ROOT="${FIXED}" CHECKLIST="${CHECKLIST}" PROMPT_FILES="${PROMPTS}" DYN_ROOT="${DYN}" GPU_IDS="${ALL_GPUS}" \
      bash "${S}/run_univ_dynamic.sh" "${m}" all
  done
}

quality() {
  say "quality layer + v2 judder/flicker checks (thresholds from the main experiment's FULL clips)"
  local m
  for m in plan run v2; do
    FIXED_ROOT="${FIXED}" Q_ROOT="${Q}" GPU_IDS="${ALL_GPUS}" CALIB_FIXED="${OUT}/fixed_metrics_v1/fixed_by_video.csv" \
      bash "${S}/run_univ_quality.sh" "${m}"
  done
}

route() {
  say "routing: table, prompt vectors, CV comparison"
  local env=(FIXED_ROOT="${FIXED}" DYN_CSV="${DYN}/dynamic_by_video.csv" Q_CSV="${Q}_v2/quality_by_video.csv"
             ROUTE_ROOT="${ROUTE}" PROMPT_FILES="${PROMPTS}")
  env "${env[@]}" bash "${S}/run_univ_route.sh" table
  [[ -f "${ROUTE}/emb_t5.npz" ]] || env "${env[@]}" GPU="${GPUS[0]}" bash "${S}/run_univ_route.sh" embed-t5
  [[ -f "${ROUTE}/emb_qwen.npz" ]] || env "${env[@]}" GPU="${GPUS[1]}" bash "${S}/run_univ_route.sh" embed-qwen
  env "${env[@]}" bash "${S}/run_univ_route.sh" embed-eval
}

status() {
  echo "prompts:   $(wc -l < "${PROMPTS}" 2>/dev/null || echo 0)"
  echo "checklist: $(ls "${GEN}/raw" 2>/dev/null | wc -l) generated, $(wc -l < "${CHECKLIST}" 2>/dev/null || echo 0) valid"
  [[ -f "${P}/plan.json" ]] && PILOT_CONFIG="${CFG}/published_wan21_platform_natural.json" PUBLISHED_WAN21_ROOT="${P}" \
    bash "${S}/run_univ_published_wan21_8gpu.sh" status 2>&1 | tail -3 | sed 's/^/platform  /'
  [[ -f "${T}/plan.json" ]] && TWOSTAGE_CONFIG="${CFG}/published_wan21_twostage_natural.json" TWOSTAGE_ROOT="${T}" \
    REFERENCE_ROOT="${P}" bash "${S}/run_univ_twostage_wan21_8gpu.sh" status 2>&1 | sed 's/^/twostage  /'
  echo "fixed:     $(ls "${FIXED}/cache" 2>/dev/null | wc -l) clips"
  echo "dynamic:   vlm $(ls "${DYN}/vlm" 2>/dev/null | wc -l), pair $(ls "${DYN}/pair" 2>/dev/null | wc -l)"
  echo "quality:   vlm $(ls "${Q}/vlm" 2>/dev/null | wc -l), pair $(ls "${Q}/pair" 2>/dev/null | wc -l)"
  ls "${ROUTE}"_embed_analysis.tgz 2>/dev/null || true
  return 0
}

unattended() {
  # Each stage in its own bash so set -e holds inside it; a failure stops the chain (every later stage needs it).
  local stage
  for stage in checklist platform twostage fixed dynamic quality route; do
    bash "$0" "${stage}" || { say "STAGE FAILED: ${stage}"; status; exit 1; }
  done
  status
  local bundle="${OUT}/natural_v1_results.tgz" files=()
  mapfile -t files < <(cd "${OUT}" && ls -1 published_wan21_platform_natural/exports/*.tgz published_wan21_twostage_natural/exports/*.tgz \
    requirements_natural/checklist.jsonl requirements_natural/collect.txt fixed_metrics_natural_analysis.tgz \
    dynamic_natural_analysis.tgz quality_natural_analysis.tgz quality_natural_v2/quality_by_video.csv \
    quality_natural_v2/report.txt route_natural_embed_analysis.tgz natural_v1.log 2>/dev/null || true)
  tar czf "${bundle}" -C "${OUT}" "${files[@]}" && echo "SEND THIS: ${bundle}"
  say "NATURAL BATCH DONE"
}

case "${STAGE}" in
  all) unattended ;;
  env) envcheck ;;
  prompts|checklist|platform|twostage|fixed|dynamic|quality|route|status) "${STAGE}" ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
