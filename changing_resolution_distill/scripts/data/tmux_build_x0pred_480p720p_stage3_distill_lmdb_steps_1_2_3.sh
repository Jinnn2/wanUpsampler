#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
USER_MODEL_ROOT="${MODEL_ROOT:-}"
STEPS="${STEPS:-1,2,3}"
STEP_TAG="${STEPS//,/_}"
SESSION_NAME="${SESSION_NAME:-wan_cr_distill_stage3_x0pred_steps_${STEP_TAG}_build}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
START_OFFSET="${START_OFFSET:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OVERWRITE="${OVERWRITE:-0}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
CR_DISTILL_MODEL_ROOT="${CR_DISTILL_MODEL_ROOT:-/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CR_DISTILL_MODEL_ID="${CR_DISTILL_MODEL_ID:-lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG:-14b_cfgdistill}"
MODEL_ROOT="${USER_MODEL_ROOT:-${CR_DISTILL_MODEL_ROOT}}"
CR_STAGE2_LMDB_DIR="${CR_STAGE2_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CR_DISTILL_STAGE3_X0PRED_CONFIG="${CR_DISTILL_STAGE3_X0PRED_CONFIG:-${PROJECT_ROOT}/changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_480p.json}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500 250}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5}"
GUIDE_SCALE="${GUIDE_SCALE:-6}"
BASE_SEED="${BASE_SEED:-9400}"
MODE="${MODE:-lightx2v_distill}"
PRECISION="${PRECISION:-bf16}"

TMUX_LOG_DIR="${TMUX_LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${TMUX_LOG_DIR}/build_x0pred_480p720p_stage3_distill_steps_${STEP_TAG}.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${TMUX_LOG_DIR}/run_x0pred_480p720p_stage3_distill_steps_${STEP_TAG}_build.tmux.sh}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run the per-step build script directly." >&2
  exit 1
fi

mkdir -p "${TMUX_LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Run log: ${RUN_LOG}"
  exit 0
fi

cat >"${RUN_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT}"

export PROJECT_ROOT="${PROJECT_ROOT}"
export TOTAL_SAMPLES="${TOTAL_SAMPLES}"
export START_OFFSET="${START_OFFSET}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export OVERWRITE="${OVERWRITE}"
export LIGHTX2V_REPO="${LIGHTX2V_REPO}"
export CR_DISTILL_MODEL_ROOT="${CR_DISTILL_MODEL_ROOT}"
export CR_DISTILL_MODEL_ID="${CR_DISTILL_MODEL_ID}"
export CR_DISTILL_STAGE3_TAG="${CR_DISTILL_STAGE3_TAG}"
export MODEL_ROOT="${MODEL_ROOT}"
export CR_STAGE2_LMDB_DIR="${CR_STAGE2_LMDB_DIR}"
export CR_DISTILL_STAGE3_X0PRED_CONFIG="${CR_DISTILL_STAGE3_X0PRED_CONFIG}"
export DENOISING_STEP_LIST="${DENOISING_STEP_LIST}"
export SAMPLE_SHIFT="${SAMPLE_SHIFT}"
export GUIDE_SCALE="${GUIDE_SCALE}"
export BASE_SEED="${BASE_SEED}"
export MODE="${MODE}"
export PRECISION="${PRECISION}"

echo "tmux session: ${SESSION_NAME}"
echo "project      : ${PROJECT_ROOT}"
echo "steps        : ${STEPS}"
echo "total_samples: ${TOTAL_SAMPLES}"
echo "start_offset : ${START_OFFSET}"
echo "gpu          : ${CUDA_VISIBLE_DEVICES}"
echo "stage3_tag   : ${CR_DISTILL_STAGE3_TAG}"
echo "distill_id   : ${CR_DISTILL_MODEL_ID}"
echo "run_log      : ${RUN_LOG}"

IFS=',' read -r -a STEP_LIST <<< "${STEPS}"
for step in "\${STEP_LIST[@]}"; do
  step="\$(echo "\${step}" | xargs)"
  if [[ -z "\${step}" ]]; then
    continue
  fi
  export HANDOFF_STEP="\${step}"
  export MAX_SAMPLES="${TOTAL_SAMPLES}"
  export SAMPLE_OFFSET="${START_OFFSET}"
  export CR_DISTILL_STAGE3_LMDB_DIR="${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_${CR_DISTILL_STAGE3_TAG}_step\${step}"

  echo "===== build Stage 3 distill x0-pred LMDB handoff step \${step} ====="
  bash changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb.sh
done

echo "All Stage 3 distill x0-pred LMDB builds finished for steps: ${STEPS}"
EOF

chmod +x "${RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash '${RUN_SCRIPT}' 2>&1 | tee '${RUN_LOG}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Run log: ${RUN_LOG}"
