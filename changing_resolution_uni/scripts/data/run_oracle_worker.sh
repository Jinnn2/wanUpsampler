#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LIGHTX2V_REPO="${LIGHTX2V_REPO:-/mnt/afs_2/houze/LightX2V}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${CR_STAGE2_368X640_720X1248_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb}/latest.pt}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml}"
PROMPTS_FILE="${PROMPTS_FILE:-${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset}"
T5_EMBED_DIR="${T5_EMBED_DIR:-${OUT_ROOT}/t5_embeddings}"
VBENCH_ROOT="${VBENCH_ROOT:-}"
ENABLE_INLINE_VBENCH="${ENABLE_INLINE_VBENCH:-0}"

PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
LIMIT="${LIMIT:-500}"
PROTOCOL_PROMPT_OFFSET="${PROTOCOL_PROMPT_OFFSET:-${PROMPT_OFFSET}}"
PROTOCOL_PROMPT_LIMIT="${PROTOCOL_PROMPT_LIMIT:-${LIMIT}}"
SEEDS="${SEEDS:-42 100 2024}"
CANDIDATE_STEPS="${CANDIDATE_STEPS:-30 35 40 41 42 43 44 45 46 47 48 49 50}"
INFER_STEPS="${INFER_STEPS:-50}"
LR_H="${LR_H:-368}"
LR_W="${LR_W:-640}"
HR_H="${HR_H:-720}"
HR_W="${HR_W:-1248}"
NUM_FRAMES="${NUM_FRAMES:-81}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_VIDEOS="${CLEAN_VIDEOS:-0}"
INCLUDE_NATIVE_HR="${INCLUDE_NATIVE_HR:-1}"
SAVE_LATENTS="${SAVE_LATENTS:-0}"
LATENT_SAVE_DTYPE="${LATENT_SAVE_DTYPE:-fp16}"

if [[ "${INCLUDE_NATIVE_HR}" != "0" && "${INCLUDE_NATIVE_HR}" != "1" ]]; then
  echo "INCLUDE_NATIVE_HR must be 0 or 1, got: ${INCLUDE_NATIVE_HR}" >&2
  exit 2
fi
if [[ "${SAVE_LATENTS}" != "0" && "${SAVE_LATENTS}" != "1" ]]; then
  echo "SAVE_LATENTS must be 0 or 1, got: ${SAVE_LATENTS}" >&2
  exit 2
fi
if [[ "${LATENT_SAVE_DTYPE}" != "fp16" && "${LATENT_SAVE_DTYPE}" != "bf16" && "${LATENT_SAVE_DTYPE}" != "fp32" ]]; then
  echo "LATENT_SAVE_DTYPE must be fp16, bf16, or fp32, got: ${LATENT_SAVE_DTYPE}" >&2
  exit 2
fi
if [[ "${ENABLE_INLINE_VBENCH}" != "0" && "${ENABLE_INLINE_VBENCH}" != "1" ]]; then
  echo "ENABLE_INLINE_VBENCH must be 0 or 1, got: ${ENABLE_INLINE_VBENCH}" >&2
  exit 2
fi

export LIGHTX2V_REPO MODEL_ROOT PROJECT_ROOT
export INFER_STEPS NUM_FRAMES HR_H HR_W LR_H LR_W STAGE2_CHECKPOINT STAGE2_TRAIN_CONFIG
export PYTHONPATH="${LIGHTX2V_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[Worker GPU=${GPU_ID}] offset=${PROMPT_OFFSET}, limit=${LIMIT}, seeds='${SEEDS}'"
echo "  Prompts: ${PROMPTS_FILE}"
echo "  Output : ${OUT_ROOT}"
echo "  DryRun : ${DRY_RUN}"
echo "  Native : ${INCLUDE_NATIVE_HR}"
echo "  Latents: ${SAVE_LATENTS} (${LATENT_SAVE_DTYPE})"

mkdir -p "${OUT_ROOT}"

# ── 1. Batch Branch Generation (Model is loaded ONCE per seed across all prompts) ──
for seed in ${SEEDS}; do
  seed_out="${OUT_ROOT}/raw_samples/seed_${seed}"
  mkdir -p "${seed_out}/configs"
  
  config_json="${seed_out}/configs/stage2_branch_config.json"
  python - "${config_json}" "${CANDIDATE_STEPS%% *}" <<'PY'
import json, os, sys
from pathlib import Path
path, first_step = sys.argv[1], int(sys.argv[2])
cfg = {
    "infer_steps": int(os.environ["INFER_STEPS"]),
    "target_video_length": int(os.environ["NUM_FRAMES"]),
    "text_len": 512,
    "target_height": int(os.environ["HR_H"]),
    "target_width": int(os.environ["HR_W"]),
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": 6.0,
    "sample_shift": 8.0,
    "enable_cfg": True,
    "cpu_offload": False,
    "feature_caching": "NoCaching",
    "parallel": False,
    "changing_resolution": True,
    "resolution_rate": [int(os.environ["LR_H"]) / int(os.environ["HR_H"])],
    "wan_lowres_latent_size": [int(os.environ["LR_H"]) // 8, int(os.environ["LR_W"]) // 8],
    "changing_resolution_steps": [first_step],
    "wan_clean_resizer_repo": os.environ["PROJECT_ROOT"],
    "wan_clean_resizer_ckpt": os.environ.get("STAGE2_CHECKPOINT", ""),
    "wan_clean_resizer_train_config": os.environ.get("STAGE2_TRAIN_CONFIG", ""),
    "wan_clean_resizer_model_class": "stage2",
    "wan_clean_resizer_use_ema": False,
}
output = Path(path)
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
os.replace(temporary, output)
PY

  if [[ "${DRY_RUN}" != "1" ]]; then
    echo "[Worker GPU=${GPU_ID}] Running streaming branch generation: seed=${seed}, prompts=${LIMIT} (Single model load)"
    infer_script="${PROJECT_ROOT}/changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_oracle_branch_infer.py"
    
    infer_args=(
      --seed "${seed}"
      --model_cls "wan2.1_clean_resizer_bridge"
      --task t2v
      --model_path "${MODEL_ROOT}"
      --config_json "${config_json}"
      --prompts_file "${PROMPTS_FILE}"
      --prompt-offset "${PROMPT_OFFSET}"
      --limit "${LIMIT}"
      --protocol-prompt-offset "${PROTOCOL_PROMPT_OFFSET}"
      --protocol-prompt-limit "${PROTOCOL_PROMPT_LIMIT}"
      --change-steps "${CANDIDATE_STEPS}"
      --infer-steps "${INFER_STEPS}"
      --lr-height "${LR_H}"
      --lr-width "${LR_W}"
      --hr-height "${HR_H}"
      --hr-width "${HR_W}"
      --lr-latent-height "$((LR_H / 8))"
      --lr-latent-width "$((LR_W / 8))"
      --target_video_length "${NUM_FRAMES}"
      --out-root "${seed_out}"
      --execution-mode "branch"
    )
    if [[ "${INCLUDE_NATIVE_HR}" == "1" ]]; then
      infer_args+=(--include-native-hr)
    fi
    if [[ "${SAVE_LATENTS}" == "1" ]]; then
      infer_args+=(--save-latents --latent-save-dtype "${LATENT_SAVE_DTYPE}")
    fi
    if [[ "${SKIP_EXISTING}" == "1" ]]; then
      infer_args+=(--skip-existing)
    fi
    
    python "${infer_script}" "${infer_args[@]}"
  fi
done

# ── 2. Index Trajectories & Compute Utilities ────────────────────────────────
echo "[Worker GPU=${GPU_ID}] Packaging trajectory dataset records..."
py_args=(
  --prompts_file "${PROMPTS_FILE}"
  --out_root "${OUT_ROOT}"
  --prompt_offset "${PROMPT_OFFSET}"
  --limit "${LIMIT}"
  --seeds ${SEEDS}
  --candidate_steps ${CANDIDATE_STEPS}
  --infer_steps "${INFER_STEPS}"
  --primary_lambda "${PRIMARY_LAMBDA}"
)

if [[ -d "${T5_EMBED_DIR}" ]]; then
  py_args+=(--t5_embed_dir "${T5_EMBED_DIR}")
fi
if [[ "${ENABLE_INLINE_VBENCH}" == "1" && -n "${VBENCH_ROOT}" && -d "${VBENCH_ROOT}" ]]; then
  py_args+=(--vbench_root "${VBENCH_ROOT}")
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  py_args+=(--skip_existing)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  py_args+=(--dry_run)
fi
if [[ "${CLEAN_VIDEOS}" == "1" ]]; then
  py_args+=(--clean_videos_after_eval)
fi

python "${SCRIPT_DIR}/build_oracle_trajectory_dataset.py" "${py_args[@]}"
