#!/usr/bin/env bash

# Local machine paths for /mnt/afs_2/houze.
# Scripts source this file by default. Override with:
#   PATH_CONFIG=/path/to/local_paths.sh bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check

JIN_ROOT="${JIN_ROOT:-/mnt/afs_2/houze}"

PROJECT_ROOT="${PROJECT_ROOT:-${JIN_ROOT}/wanUpsampler}"
WAN_REPO="${WAN_REPO:-${JIN_ROOT}/Wan2.1}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-${JIN_ROOT}/LightX2V}"

MODEL_ROOT="${MODEL_ROOT:-${JIN_ROOT}/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"

# Stage 2 changing_resolution clean-latent paths.
CR_RAW_VIDEO_DIR="${CR_RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p}"
CR_RAW_VIDEO_DIR_1K="${CR_RAW_VIDEO_DIR_1K:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p_1k}"
CR_LMDB_DIR="${CR_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CR_STAGE2_CONFIG="${CR_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml}"
CR_STAGE2_OUT_DIR="${CR_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p_stage2_lmdb}"
CR_GENERATE_CONFIG="${CR_GENERATE_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p.json}"
CR_PROMPTS_FILE="${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}"
CR_HF_PROMPTS_FILE="${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}"
CR_STAGE2_OPERATOR_COMPARE_CKPT="${CR_STAGE2_OPERATOR_COMPARE_CKPT:-${CR_STAGE2_OUT_DIR}/latest.pt}"
CR_STAGE2_OPERATOR_COMPARE_DIR="${CR_STAGE2_OPERATOR_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_operator_compare_stage2}"
CR_STAGE2_CHAIN_COMPARE_CKPT="${CR_STAGE2_CHAIN_COMPARE_CKPT:-${CR_STAGE2_OUT_DIR}/latest.pt}"
CR_STAGE2_CHAIN_COMPARE_DIR="${CR_STAGE2_CHAIN_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_chain_ab_stage2}"

# Stage 3 x0-pred clean-latent paths.
CR_STAGE3_DENOISE_STEP="${DENOISE_STEP:-45}"
CR_STAGE3_HR_TARGET_MODE="${HR_TARGET_MODE:-x0_pred}"
case "${CR_STAGE3_HR_TARGET_MODE}" in
  x0_pred)
    CR_STAGE3_DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_x0predhr_step${CR_STAGE3_DENOISE_STEP}"
    CR_STAGE3_DEFAULT_OUT_NAME="changing_resolution_x0pred_480p720p_stage3_x0predhr_step${CR_STAGE3_DENOISE_STEP}_lmdb"
    ;;
  clean)
    CR_STAGE3_DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_cleanhr_step${CR_STAGE3_DENOISE_STEP}"
    CR_STAGE3_DEFAULT_OUT_NAME="changing_resolution_x0pred_480p720p_stage3_cleanhr_step${CR_STAGE3_DENOISE_STEP}_lmdb"
    ;;
  *)
    CR_STAGE3_DEFAULT_LMDB_NAME="lmdb_x0pred_480p720p_stage3_${CR_STAGE3_HR_TARGET_MODE}_step${CR_STAGE3_DENOISE_STEP}"
    CR_STAGE3_DEFAULT_OUT_NAME="changing_resolution_x0pred_480p720p_stage3_${CR_STAGE3_HR_TARGET_MODE}_step${CR_STAGE3_DENOISE_STEP}_lmdb"
    ;;
esac
CR_STAGE3_LMDB_DIR="${CR_STAGE3_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/${CR_STAGE3_DEFAULT_LMDB_NAME}}"
CR_STAGE3_CONFIG="${CR_STAGE3_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_x0pred_480p_to_720p_lmdb_stage3.yaml}"
CR_STAGE3_OUT_DIR="${CR_STAGE3_OUT_DIR:-${PROJECT_ROOT}/outputs/${CR_STAGE3_DEFAULT_OUT_NAME}}"
CR_STAGE3_CHANGE_STEP_SWEEP_CKPT="${CR_STAGE3_CHANGE_STEP_SWEEP_CKPT:-${CR_STAGE3_OUT_DIR}/latest.pt}"
CR_STAGE3_CHANGE_STEP_SWEEP_DIR="${CR_STAGE3_CHANGE_STEP_SWEEP_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_stage3_change_step_sweep}"
