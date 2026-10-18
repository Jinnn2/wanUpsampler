#!/usr/bin/env bash
# Official DVG vs FULL on Wan2.1-T2V-1.3B (832x480x81), then VBench and requirement scoring.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
GEN_PYTHON="${GEN_PYTHON:-${REPO_ROOT}/.venv_published_wan21/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/bin/python}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
DVG_ROOT="${DVG_ROOT:-/mnt/afs_2/houze/DVG}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
VBENCH_PROMPTS="${VBENCH_PROMPTS:-${VBENCH_ROOT}/prompts/all_dimension.txt}"
VBENCH_EXT_PROMPTS="${VBENCH_EXT_PROMPTS:-${VBENCH_ROOT}/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt}"
SETS="${SETS:-main,repro}"
PROMPTS="${PROMPTS:-${REPO_ROOT}/prompts/univ_controlled_factor_v1.jsonl}"
STUDY_ROOT="${STUDY_ROOT:-${REPO_ROOT}/outputs/dvg_wan21_study_v1}"
REQ_ROOT="${REQ_ROOT:-${REPO_ROOT}/outputs/dvg_wan21_requirements_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRIVER="${REPO_ROOT}/UNIV_adaptor/scripts/data/dvg_wan21_study.py"
REQ="${REPO_ROOT}/UNIV_adaptor/scripts/run_univ_controlled_factor_requirements.sh"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
cd "${REPO_ROOT}"
# VBench runs with its own cwd, so study paths given relative to the repo must be absolute.
STUDY_ROOT="$(realpath -m "${STUDY_ROOT}")"
REQ_ROOT="$(realpath -m "${REQ_ROOT}")"

case "${MODE}" in
  plan)
    "${BASE_PYTHON}" "${DRIVER}" plan --out "${STUDY_ROOT}" --prompts "${PROMPTS}" --vbench-prompts "${VBENCH_PROMPTS}" \
      --vbench-ext-prompts "${VBENCH_EXT_PROMPTS}" --sets "${SETS}"
    ;;
  smoke)
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${GEN_PYTHON}" "${DRIVER}" smoke --out "${STUDY_ROOT}" \
      --dvg-root "${DVG_ROOT}" --ckpt-dir "${MODEL_ROOT}"
    ;;
  official-check)
    # Re-run short VBench prompts through DVG's own Wan2.1/generate.py (seed 42, D=0.5), then compare
    # frame-by-frame with our receipted videos. FA3 is disabled the same way as in our driver.
    check="${STUDY_ROOT}/official_check"
    mkdir -p "${check}"
    printf '%s\n' '{"prompt": "downtown"}' '{"prompt": "shower"}' '{"prompt": "A person is clapping"}' \
      '{"prompt": "a bench"}' '{"prompt": "a horse galloping across an open field"}' \
      '{"prompt": "A person is clay pottery making"}' '{"prompt": "a truck slowing down to stop"}' > "${check}/prompts.jsonl"
    (cd "${DVG_ROOT}" && CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONPATH="${DVG_ROOT}:${DVG_ROOT}/Wan2.1" "${GEN_PYTHON}" -c '
import runpy, sys
from wan.modules import attention
attention.FLASH_ATTN_3_AVAILABLE = False
sys.argv = ["generate.py"] + sys.argv[1:]
runpy.run_path("Wan2.1/generate.py", run_name="__main__")
' --task t2v-1.3B --size '832*480' --ckpt_dir "${MODEL_ROOT}" --prompt_file "${check}/prompts.jsonl" \
      --output_dir "${check}/dvg050" --sample_steps 50 --sample_shift 8.0 --sample_guide_scale 6.0 --frame_num 81 \
      --base_seed 42 --enable_tae false --enable_torch_compile false --ulysses_size 1 --offload_model false \
      --enable_dvg --dvg_budget 0.5) 2>&1 | tee "${check}/run.log"
    "${GEN_PYTHON}" - "${check}/dvg050" "${STUDY_ROOT}/videos" <<'PY'
import sys, glob, os, numpy as np, imageio
official, ours = sys.argv[1], sys.argv[2]
pairs = {"00000": "repro_v0804_s42", "00001": "repro_v0844_s42", "00002": "repro_v0161_s42", "00003": "repro_v0342_s42"}
def load(p):
    return np.stack([f for f in imageio.get_reader(p)]).astype(np.float32)
for path in sorted(glob.glob(f"{official}/*.mp4")):
    idx = os.path.basename(path)[:5]
    a = load(path)
    line = f"{os.path.basename(path)}: frames={len(a)} mean|dframe|={np.abs(np.diff(a, axis=0)).mean():.2f}"
    if idx in pairs:
        b = load(f"{ours}/{pairs[idx]}__DVG_D050.mp4")
        line += f" | vs ours {pairs[idx]}: mean|diff|={np.abs(a - b).mean():.2f} (0 = identical)"
    print(line)
PY
    tar czf "${check}.tgz" -C "${STUDY_ROOT}" official_check
    echo "Download: ${check}.tgz"
    ;;
  generate)
    mkdir -p "${STUDY_ROOT}/logs"
    pids=()
    for i in "${!GPUS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${GEN_PYTHON}" "${DRIVER}" generate --out "${STUDY_ROOT}" \
        --dvg-root "${DVG_ROOT}" --ckpt-dir "${MODEL_ROOT}" --shard-index "${i}" --shard-count "${#GPUS[@]}" \
        > "${STUDY_ROOT}/logs/generate_${i}.log" 2>&1 &
      pids+=("$!")
    done
    trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
    failed=0
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "worker ${i} failed: ${STUDY_ROOT}/logs/generate_${i}.log" >&2; failed=1; }
    done
    exit "${failed}"
    ;;
  status)
    "${BASE_PYTHON}" "${DRIVER}" status --out "${STUDY_ROOT}"
    ;;
  score)
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VBENCH_PYTHON}" "${DRIVER}" score --out "${STUDY_ROOT}" \
      --vbench-root "${VBENCH_ROOT}" --vbench-python "${VBENCH_PYTHON}" --ngpus "${#GPUS[@]}"
    ;;
  requirements)
    "${BASE_PYTHON}" "${REPO_ROOT}/UNIV_adaptor/scripts/data/score_controlled_factor_requirements.py" plan \
      --dvg-study "${STUDY_ROOT}" --checklist "${REPO_ROOT}/prompts/univ_controlled_factor_v1_requirements.jsonl" \
      --out "${REQ_ROOT}"
    REQ_ROOT="${REQ_ROOT}" QUALITY_CSV="${STUDY_ROOT}/quality_by_video.csv" bash "${REQ}" signals
    REQ_ROOT="${REQ_ROOT}" QUALITY_CSV="${STUDY_ROOT}/quality_by_video.csv" bash "${REQ}" vlm
    REQ_ROOT="${REQ_ROOT}" QUALITY_CSV="${STUDY_ROOT}/quality_by_video.csv" bash "${REQ}" report
    ;;
  export)
    name="$(basename "${STUDY_ROOT}")"
    files=()
    for f in plan.json receipts vbench_scores.json quality_by_video.csv repro_by_video.csv repro_ext_by_video.csv \
             repro_summary.md repro_ext_summary.md; do
      [[ -e "${STUDY_ROOT}/${f}" ]] && files+=("${name}/${f}")
    done
    tar czf "${STUDY_ROOT}_analysis.tgz" -C "$(dirname "${STUDY_ROOT}")" "${files[@]}"
    echo "Download: ${STUDY_ROOT}_analysis.tgz and ${REQ_ROOT}_analysis.tgz"
    ;;
  *)
    echo "Usage: bash $0 {plan|smoke|official-check|generate|status|score|requirements|export}"
    echo "Clone DVG first: git clone https://github.com/DarrenZheng303/DVG.git ${DVG_ROOT}"
    echo "Plan: 64 dev prompts x 3 seeds x {FULL, DVG 0.7, DVG 0.5} + 48 VBench prompts x {FULL, DVG 0.5}."
    echo "Extended-prompt control (same 48 VBench indices, VBench's Wan2.1 prompt extension), in its own root:"
    echo "  STUDY_ROOT=outputs/dvg_wan21_vbench_ext_v1 SETS=repro_ext bash $0 plan   # then generate/score/export"
    ;;
esac
